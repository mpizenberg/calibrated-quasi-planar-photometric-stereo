module Icon exposing (arrowDown, arrowLeftCircle, arrowRightCircle, boundingBox, check, fileText, github, image, logIn, maximize, move, search, slash, sunset, trash, zoomFit, zoomIn, zoomOut)

import Element exposing (Element)
import FeatherIcons
import Svg exposing (Svg, svg)
import Svg.Attributes exposing (..)



-- Designed by Matthieu ####################################


toElement : List (Svg msg) -> Float -> Element msg
toElement icon size =
    svg (width (String.fromFloat size) :: height (String.fromFloat size) :: defaultAttributes) icon
        |> Element.html


defaultAttributes : List (Svg.Attribute msg)
defaultAttributes =
    [ fill "none"
    , stroke "currentColor"
    , strokeLinecap "round"
    , strokeLinejoin "round"
    , strokeWidth "2"
    , viewBox "0 0 24 24"
    ]


zoomFit : Float -> Element msg
zoomFit =
    toElement
        [ Svg.circle [ cx "11", cy "11", r "8" ] []
        , Svg.line [ x1 "21", y1 "21", x2 "16.65", y2 "16.65" ] []
        , Svg.path [ d "M 6 8 v 6 h 10 v -6 h -10" ] []
        ]


boundingBox : Float -> Element msg
boundingBox =
    toElement
        [ Svg.path [ d "M 23 17 h -6 m -3 0 H 4 V 7 H 20 V 11 m 0 3 v 6" ] []
        ]



-- Feather icons


featherIcon : FeatherIcons.Icon -> Float -> Element msg
featherIcon icon size =
    Element.html (FeatherIcons.toHtml [] (FeatherIcons.withSize size icon))


github : Float -> Element msg
github =
    featherIcon FeatherIcons.github


image : Float -> Element msg
image =
    featherIcon FeatherIcons.image


fileText : Float -> Element msg
fileText =
    featherIcon FeatherIcons.fileText


arrowDown : Float -> Element msg
arrowDown =
    featherIcon FeatherIcons.arrowDown


arrowLeftCircle : Float -> Element msg
arrowLeftCircle =
    featherIcon FeatherIcons.arrowLeftCircle


arrowRightCircle : Float -> Element msg
arrowRightCircle =
    featherIcon FeatherIcons.arrowRightCircle

check : Float -> Element msg
check =
    featherIcon FeatherIcons.check


logIn: Float -> Element msg
logIn =
    featherIcon FeatherIcons.logIn

move : Float -> Element msg
move =
    featherIcon FeatherIcons.move

search : Float -> Element msg
search =
    featherIcon FeatherIcons.search

slash : Float -> Element msg
slash =
    featherIcon FeatherIcons.slash

sunset : Float -> Element msg
sunset =
    featherIcon FeatherIcons.sunset

trash : Float -> Element msg
trash =
    featherIcon FeatherIcons.trash2


zoomIn : Float -> Element msg
zoomIn =
    featherIcon FeatherIcons.zoomIn


zoomOut : Float -> Element msg
zoomOut =
    featherIcon FeatherIcons.zoomOut


maximize : Float -> Element msg
maximize =
    featherIcon FeatherIcons.maximize
