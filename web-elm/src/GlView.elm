module GlView exposing (InnerMsg, Model, Msg(..), init, update, view)

import Angle exposing (Angle)
import Camera3d exposing (Camera3d)
import Direction3d
import Frame3d
import Html exposing (Html)
import Html.Attributes as HA exposing (height, style, width)
import Html.Events as HE
import Html.Events.Extra.Mouse as Mouse
import Html.Events.Extra.Wheel as Wheel
import Json.Decode as Decode exposing (Decoder)
import Length exposing (Length, Meters)
import Math.Matrix4 exposing (Mat4)
import Math.Vector3 exposing (Vec3, vec3)
import Point3d exposing (Point3d)
import Quantity
import Task exposing (Task)
import Vector3d
import Viewpoint3d
import WebGL exposing (Mesh, Shader)
import WebGL.Matrices
import WebGL.Texture as Texture exposing (Texture)


type Model
    = Landing
    | LoadingTexture
    | TransferringTextureToWebGL
    | ErrorLoadingTexture Texture.Error
    | Rendering RenderingModel


init : Model
init =
    Landing


type alias RenderingModel =
    { depthMap : Texture
    , size : ( Int, Int )
    , xyScale : Float
    , mesh : Mesh Vertex
    , currentTime : Float
    , controlling : CameraControl
    , camera : OrbitCamera
    , lighting : Lighting
    , depthScale : Float
    }


type alias Lighting =
    { azimuth : Angle
    , elevation : Angle
    }


initialLighting : Lighting
initialLighting =
    { azimuth = Quantity.zero
    , elevation = Angle.degrees 90
    }


type alias OrbitCamera =
    { focalPoint : Point3d Meters ()
    , azimuth : Angle
    , elevation : Angle
    , distance : Length
    }


type CameraControl
    = NoControl
    | Orbiting
    | Panning


initialOrbitCamera : ( Float, Float ) -> OrbitCamera
initialOrbitCamera ( targetX, targetY ) =
    { focalPoint = Point3d.xyz (Length.meters targetX) (Length.meters targetY) Quantity.zero
    , azimuth = Angle.degrees -90
    , elevation = Angle.degrees 60
    , distance = Length.meters 2
    }



-- Update


movementDecoder : Decoder ( Float, Float )
movementDecoder =
    Decode.map2 (\a b -> ( a, b ))
        (Decode.field "movementX" Decode.float)
        (Decode.field "movementY" Decode.float)


type Msg
    = InnerMsg InnerMsg
    | SetImageUrl String
    | MouseDown Mouse.Event
      -- Currently, this corresponds to the "movementX" and Y properties of the event
    | MouseMove ( Float, Float )
    | MouseUp


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    let
        innerMsg =
            case msg of
                InnerMsg inner ->
                    inner

                SetImageUrl url ->
                    UrlGenerated url

                MouseDown event ->
                    MouseDownInner event

                MouseMove mm ->
                    MouseMoveInner mm

                MouseUp ->
                    MouseUpInner

        ( newModel, cmds ) =
            updateInner innerMsg model
    in
    ( newModel, Cmd.map InnerMsg cmds )


type InnerMsg
    = UrlGenerated String
    | TextureLoaded (Result Texture.Error Texture)
    | ClickedSelectImageButton
      -- Camera
    | ZoomIn
    | ZoomOut
    | MouseDownInner Mouse.Event
    | MouseMoveInner ( Float, Float )
    | MouseUpInner
      -- Lighting
    | ChangeLightAzimuth Float
    | ChangeLightElevation Float
      -- Depth scale
    | ChangeDepthScale Float


loadTexture : String -> Task Texture.Error Texture
loadTexture =
    Texture.loadWith
        { magnify = Texture.linear
        , minify = Texture.nearest
        , horizontalWrap = Texture.clampToEdge
        , verticalWrap = Texture.clampToEdge
        , flipY = True
        }


updateInner : InnerMsg -> Model -> ( Model, Cmd InnerMsg )
updateInner msg model =
    case ( msg, model ) of
        ( UrlGenerated url, _ ) ->
            ( TransferringTextureToWebGL, Task.attempt TextureLoaded (loadTexture url) )

        ( TextureLoaded (Err err), _ ) ->
            ( ErrorLoadingTexture err, Cmd.none )

        ( TextureLoaded (Ok texture), _ ) ->
            let
                ( w, h ) =
                    Texture.size texture
            in
            ( Rendering
                { depthMap = texture
                , mesh = WebGL.indexedTriangles (meshVertices w h) (meshIndices w h)
                , xyScale = min (1 / toFloat w) (1 / toFloat h)
                , size = ( w, h )
                , currentTime = 0
                , controlling = NoControl
                , camera = initialOrbitCamera (centerTarget ( w, h ))
                , lighting = initialLighting
                , depthScale = 0.1
                }
            , Cmd.none
            )

        -- Camera
        ( ZoomIn, Rendering r ) ->
            ( Rendering { r | camera = controlZoomIn r.camera }, Cmd.none )

        ( ZoomOut, Rendering r ) ->
            ( Rendering { r | camera = controlZoomOut r.camera }, Cmd.none )

        ( MouseDownInner event, Rendering r ) ->
            ( Rendering (controlMouseDown event r), Cmd.none )

        ( MouseUpInner, Rendering r ) ->
            ( Rendering (controlMouseUp r), Cmd.none )

        ( MouseMoveInner movement, Rendering r ) ->
            ( Rendering { r | camera = controlMouseMove movement r.controlling r.camera }, Cmd.none )

        -- Lighting
        ( ChangeLightAzimuth az, Rendering r ) ->
            ( Rendering { r | lighting = changeLightAzimuth az r.lighting }, Cmd.none )

        ( ChangeLightElevation el, Rendering r ) ->
            ( Rendering { r | lighting = changeLightElevation el r.lighting }, Cmd.none )

        -- Depth scale
        ( ChangeDepthScale scale, Rendering r ) ->
            ( Rendering { r | depthScale = scale }, Cmd.none )

        _ ->
            ( model, Cmd.none )



-- Camera


controlZoomIn : OrbitCamera -> OrbitCamera
controlZoomIn camera =
    { camera | distance = Quantity.multiplyBy (21 / 29.7) camera.distance }


controlZoomOut : OrbitCamera -> OrbitCamera
controlZoomOut camera =
    { camera | distance = Quantity.multiplyBy (29.7 / 21) camera.distance }


controlMouseDown : Mouse.Event -> RenderingModel -> RenderingModel
controlMouseDown event rendering =
    let
        controlling =
            if event.keys.ctrl then
                Panning

            else
                Orbiting
    in
    { rendering | controlling = controlling }


controlMouseUp : RenderingModel -> RenderingModel
controlMouseUp rendering =
    { rendering | controlling = NoControl }


controlMouseMove : ( Float, Float ) -> CameraControl -> OrbitCamera -> OrbitCamera
controlMouseMove ( dx, dy ) controlling camera =
    case controlling of
        NoControl ->
            camera

        Orbiting ->
            orbit dx dy camera

        Panning ->
            pan dx dy camera


orbit : Float -> Float -> OrbitCamera -> OrbitCamera
orbit dx dy camera =
    let
        minElevation =
            Angle.degrees 0

        maxElevation =
            Angle.degrees 90
    in
    { focalPoint = camera.focalPoint
    , azimuth = Quantity.plus (Angle.degrees -dx) camera.azimuth
    , elevation = Quantity.clamp minElevation maxElevation (Quantity.plus (Angle.degrees dy) camera.elevation)
    , distance = camera.distance
    }


pan : Float -> Float -> OrbitCamera -> OrbitCamera
pan dx dy camera =
    let
        viewPoint =
            Viewpoint3d.orbitZ camera

        displacement =
            Vector3d.xyOn (Viewpoint3d.viewPlane viewPoint)
                (Quantity.multiplyBy (-0.001 * dx) camera.distance)
                (Quantity.multiplyBy (0.001 * dy) camera.distance)
    in
    { focalPoint = Point3d.translateBy displacement camera.focalPoint
    , azimuth = camera.azimuth
    , elevation = camera.elevation
    , distance = camera.distance
    }



-- Lighting


changeLightAzimuth : Float -> Lighting -> Lighting
changeLightAzimuth value lighting =
    { lighting | azimuth = Angle.degrees value }


changeLightElevation : Float -> Lighting -> Lighting
changeLightElevation value lighting =
    { lighting | elevation = Angle.degrees value }



-- View


view : Model -> Html Msg
view model =
    Html.map InnerMsg (viewInner model)


viewInner : Model -> Html InnerMsg
viewInner model =
    case model of
        Landing ->
            Html.div []
                [ Html.button
                    [ HE.onClick ClickedSelectImageButton ]
                    [ Html.text "Select a PNG image containing normals and depth" ]
                ]

        LoadingTexture ->
            Html.text "Loading texture ..."

        TransferringTextureToWebGL ->
            Html.text "Transferring texture to WebGL ..."

        ErrorLoadingTexture _ ->
            Html.text "X: An error occurred when loading texture"

        Rendering { depthMap, mesh, camera, lighting, depthScale, size, xyScale } ->
            Html.div []
                [ Html.button
                    [ HE.onClick ClickedSelectImageButton ]
                    [ Html.text "Select a PNG image containing normals and depth" ]
                , lightControls lighting
                , WebGL.toHtml
                    [ width 800
                    , height 800
                    , style "display" "block"
                    , Wheel.onWheel chooseZoom
                    , Mouse.onDown MouseDownInner
                    ]
                    [ WebGL.entity
                        vertexShader
                        fragmentShader
                        mesh
                        { modelViewProjection = modelViewProjection camera
                        , directionalLight = directionalLight lighting
                        , texture = depthMap
                        , scale = depthScale
                        , xyScale = xyScale
                        , width = toFloat (Tuple.first size)
                        , height = toFloat (Tuple.second size)
                        }
                    ]
                ]


directionalLight : Lighting -> Vec3
directionalLight { azimuth, elevation } =
    Direction3d.xyZ azimuth elevation
        |> Direction3d.components
        |> (\( x, y, z ) -> vec3 x y z)


lightControls : Lighting -> Html InnerMsg
lightControls lighting =
    Html.div []
        [ Html.text "Depth scale"
        , Html.input
            [ HA.type_ "number"
            , HA.min "0.01"
            , HA.max "1.00"
            , HA.step "0.01"
            , HA.placeholder "0.1"
            , HE.stopPropagationOn "change" (valueDecoder ChangeDepthScale)
            ]
            []
        , Html.p [] [ Html.text "Light direction" ]
        , Html.div []
            [ Html.text "Azimuth: 0"
            , Html.input
                [ HA.type_ "range"
                , HA.min "0"
                , HA.max "360"
                , HA.step "1"
                , HA.value (String.fromInt <| round <| Angle.inDegrees lighting.azimuth)
                , HE.stopPropagationOn "input" (valueDecoder ChangeLightAzimuth)
                ]
                []
            , Html.text "360 degrees"
            ]
        , Html.div []
            [ Html.text "Elevation: 0"
            , Html.input
                [ HA.type_ "range"
                , HA.min "0"
                , HA.max "90"
                , HA.step "1"
                , HA.value (String.fromInt <| round <| Angle.inDegrees lighting.elevation)
                , HE.stopPropagationOn "input" (valueDecoder ChangeLightElevation)
                ]
                []
            , Html.text "90 degrees"
            ]
        ]


valueDecoder : (Float -> InnerMsg) -> Decoder ( InnerMsg, Bool )
valueDecoder toMsg =
    HE.targetValue
        |> Decode.map (String.toFloat >> Maybe.withDefault 0)
        |> Decode.map toMsg
        |> Decode.map (\x -> ( x, True ))


chooseZoom : Wheel.Event -> InnerMsg
chooseZoom wheelEvent =
    if wheelEvent.deltaY > 0 then
        ZoomOut

    else
        ZoomIn



-- Camera


centerTarget : ( Int, Int ) -> ( Float, Float )
centerTarget ( w, h ) =
    let
        maxSize =
            toFloat (max w h)
    in
    ( 0.5 * toFloat w / maxSize, 0.5 * toFloat h / maxSize )


modelViewProjection : OrbitCamera -> Mat4
modelViewProjection camera =
    WebGL.Matrices.modelViewProjectionMatrix
        Frame3d.atOrigin
        (persectiveCamera camera)
        { nearClipDepth = Length.meters 0.01
        , farClipDepth = Length.meters 100
        , aspectRatio = 1
        }


persectiveCamera : OrbitCamera -> Camera3d Meters ()
persectiveCamera camera =
    Camera3d.perspective
        { viewpoint = Viewpoint3d.orbitZ camera
        , verticalFieldOfView = Angle.degrees 30
        }



-- Mesh


type alias Vertex =
    { x : Float
    , y : Float
    }


meshVertices : Int -> Int -> List Vertex
meshVertices width height =
    let
        -- We use (width-1) since elm List.range function
        -- surprisingly includes the upper bound.
        x_range =
            List.range 0 (width - 1)

        -- Same for y_range.
        y_range =
            List.range 0 (height - 1)

        makeRow : Int -> List Vertex
        makeRow y =
            List.map (\x -> Vertex (toFloat x) (toFloat y)) x_range
    in
    -- Using y for the outer loop
    -- and x for the inner loop since we are row-major
    List.concatMap makeRow y_range


meshIndices : Int -> Int -> List ( Int, Int, Int )
meshIndices width height =
    let
        -- Stop x_range at (width-2) since the last vertex of the row
        -- is already in the triangle of the previous vertex.
        x_range =
            List.range 0 (width - 2)

        -- Same for y_range.
        y_range =
            List.range 0 (height - 2)

        -- Generate the two triplets of indices for the two triangles
        -- sharing the same (x,y) top-left corner.
        trianglePairAt : Int -> Int -> List ( Int, Int, Int )
        trianglePairAt x y =
            let
                -- Convert a (x,y) subscript into its corresponding index
                -- when following the "row-major" order.
                topLeft =
                    width * y + x
            in
            -- ABG is (0, 1, 6)
            [ ( topLeft, topLeft + 1, topLeft + 1 + width )

            -- AGF is (0, 6, 5)
            , ( topLeft, topLeft + 1 + width, topLeft + width )
            ]

        makeRow : Int -> List ( Int, Int, Int )
        makeRow y =
            List.concatMap (\x -> trianglePairAt x y) x_range
    in
    -- Using y for the outer loop
    -- and x for the inner loop since we are row-major
    List.concatMap makeRow y_range



-- Shaders


type alias Uniforms =
    { modelViewProjection : Mat4
    , directionalLight : Vec3
    , texture : Texture
    , scale : Float
    , xyScale : Float
    , width : Float
    , height : Float
    }


vertexShader : Shader Vertex Uniforms { vcolor : Vec3, vnormal : Vec3 }
vertexShader =
    [glsl|

        attribute float x;
        attribute float y;
        uniform float width;
        uniform float height;
        uniform float xyScale;
        uniform float scale;
        uniform mat4 modelViewProjection;
        uniform sampler2D texture;
        varying vec3 vcolor;
        varying vec3 vnormal;

        void main () {
            float textureX = x / (width - 1.0);
            float textureY = y / (height - 1.0);
            vec4 tex = texture2D(texture, vec2(textureX, textureY));

            float nx = 2.0 * tex.x - 1.0;
            float ny = 2.0 * tex.y - 1.0;
            float nz = 2.0 * tex.z - 1.0;
            vnormal = vec3(nx, ny, nz);
            vcolor = tex.xyz;
            vec3 worldCoordinates = vec3(xyScale * x, xyScale * y, -tex.w * scale);
            gl_Position = modelViewProjection * vec4(worldCoordinates, 1.0);
        }
    |]


fragmentShader : Shader {} Uniforms { vcolor : Vec3, vnormal : Vec3 }
fragmentShader =
    [glsl|

        precision mediump float;
        uniform vec3 directionalLight;
        varying vec3 vcolor;
        varying vec3 vnormal;

        void main () {
            // normalizing the normal varying
            vec3 normal = normalize(vnormal);

            // computing directional lighting
            float intensity = dot(normal, directionalLight);

            // gl_FragColor = vec4(vcolor, 1.0);
            // gl_FragColor = vec4(intensity * vcolor, 1.0);
            gl_FragColor = vec4(intensity, intensity, intensity, 1.0);
        }

    |]
