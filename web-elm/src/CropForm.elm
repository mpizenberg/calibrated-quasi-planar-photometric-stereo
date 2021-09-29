module CropForm exposing
    ( State, withSize
    , toggle, updateLeft, updateRight, updateTop, updateBottom
    , decoded, errors
    , EventTag, boxEditor
    )

{-| Data and view of an editor for the cropped frame

@docs State, withSize

@docs toggle, updateLeft, updateRight, updateTop, updateBottom

@docs decoded, errors

@docs EventTag, boxEditor

-}

import Element exposing (Element)
import Element.Background
import Element.Border
import Element.Font
import Element.Input
import NumberInput exposing (Field, IntError)
import Style


{-| State of the different fields for a crop form.
-}
type alias State =
    { active : Bool
    , left : Field Int IntError
    , top : Field Int IntError
    , right : Field Int IntError
    , bottom : Field Int IntError
    }


{-| Initialize the crop form with 0, 0, width, height
for the left, top, right and bottom bounds
-}
withSize : Int -> Int -> State
withSize width height =
    let
        anyInt : Field Int NumberInput.IntError
        anyInt =
            NumberInput.intDefault
    in
    { active = False
    , left =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue 0
    , top =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue 0
    , right =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue width
    , bottom =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue height
    }


{-| List all errors of the crop form current state.
-}
errors : State -> List String
errors { active, left, top, right, bottom } =
    if not active then
        []

    else
        List.concat
            [ List.map (NumberInput.intErrorToString { valueName = "Left" })
                (fieldError left.decodedInput)
            , List.map (NumberInput.intErrorToString { valueName = "Top" })
                (fieldError top.decodedInput)
            , List.map (NumberInput.intErrorToString { valueName = "Right" })
                (fieldError right.decodedInput)
            , List.map (NumberInput.intErrorToString { valueName = "Bottom" })
                (fieldError bottom.decodedInput)
            ]


{-| Errors of a given field.
-}
fieldError : Result (List err) ok -> List err
fieldError result =
    case result of
        Err list ->
            list

        Ok _ ->
            []


{-| Return every field value if they are all valid.
If at least one of them isn't valid, return Nothing.
-}
decoded : State -> Maybe { left : Int, top : Int, right : Int, bottom : Int }
decoded { left, top, right, bottom } =
    case ( ( left.decodedInput, right.decodedInput ), ( top.decodedInput, bottom.decodedInput ) ) of
        ( ( Ok l, Ok r ), ( Ok t, Ok b ) ) ->
            Just { left = l, right = r, top = t, bottom = b }

        _ ->
            Nothing



-- Update


{-| Activate / deactivate the crop.
-}
toggle : Bool -> State -> State
toggle newActive data =
    { data | active = newActive }


updateLeft : String -> State -> State
updateLeft str ({ active, left, top, right, bottom } as state) =
    let
        newLeft : Field Int NumberInput.IntError
        newLeft =
            NumberInput.updateInt str left
    in
    case newLeft.decodedInput of
        Ok value ->
            { active = active
            , left = newLeft
            , top = top
            , right =
                NumberInput.setMinBound (Just value) right
                    |> NumberInput.updateInt right.input
            , bottom = bottom
            }

        Err _ ->
            { state | left = newLeft }


updateRight : String -> State -> State
updateRight str state =
    { state | right = NumberInput.updateInt str state.right }


updateTop : String -> State -> State
updateTop str ({ active, left, top, right, bottom } as state) =
    let
        newTop : Field Int NumberInput.IntError
        newTop =
            NumberInput.updateInt str top
    in
    case newTop.decodedInput of
        Ok value ->
            { active = active
            , left = left
            , top = newTop
            , right = right
            , bottom =
                NumberInput.setMinBound (Just value) bottom
                    |> NumberInput.updateInt bottom.input
            }

        Err _ ->
            { state | top = newTop }


updateBottom : String -> State -> State
updateBottom str state =
    { state | bottom = NumberInput.updateInt str state.bottom }



-- View


{-| Events dealing with changes to the input fields of the box editor.
-}
type alias EventTag msg =
    { changeLeft : String -> msg
    , changeTop : String -> msg
    , changeRight : String -> msg
    , changeBottom : String -> msg
    }


{-| The visual editor for the coordinates of each side of the cropped frame.
-}
boxEditor : EventTag msg -> State -> Element msg
boxEditor event ({ active, left, top, right, bottom } as state) =
    if not active then
        Element.none

    else
        Element.el [ Element.width Element.fill, Element.padding 4 ] <|
            Element.el
                [ Element.centerX
                , Element.centerY
                , Element.paddingXY 48 20
                , Element.Border.dashed
                , Element.Border.width 2
                , Element.onLeft
                    (Element.el (Element.moveRight 30 :: onBorderAttributes)
                        (cropField "left" event.changeLeft left)
                    )
                , Element.above
                    (Element.el (Element.moveDown 12 :: onBorderAttributes)
                        (cropField "top" event.changeTop top)
                    )
                , Element.onRight
                    (Element.el (Element.moveLeft 30 :: onBorderAttributes)
                        (cropField "right" event.changeRight right)
                    )
                , Element.below
                    (Element.el (Element.moveUp 14 :: onBorderAttributes)
                        (cropField "bottom" event.changeBottom bottom)
                    )
                ]
                (Element.el [ Element.Font.size 12 ] <|
                    case ( currentWidth state, currentHeight state ) of
                        ( Just cropWidth, Just cropHeight ) ->
                            Element.text (String.fromInt cropWidth ++ " x " ++ String.fromInt cropHeight)

                        ( Nothing, Just cropHeight ) ->
                            Element.text ("? x " ++ String.fromInt cropHeight)

                        ( Just cropWidth, Nothing ) ->
                            Element.text (String.fromInt cropWidth ++ " x ?")

                        ( Nothing, Nothing ) ->
                            Element.text "? x ?"
                )


{-| Return the width as right - left if both are valid.
-}
currentWidth : State -> Maybe Int
currentWidth { left, right } =
    case ( left.decodedInput, right.decodedInput ) of
        ( Ok l, Ok r ) ->
            Just (r - l)

        _ ->
            Nothing


{-| Return the height as bottom - top if both are valid.
-}
currentHeight : State -> Maybe Int
currentHeight { top, bottom } =
    case ( top.decodedInput, bottom.decodedInput ) of
        ( Ok t, Ok b ) ->
            Just (b - t)

        _ ->
            Nothing


{-| Common attributes for all left, top, right, bottom input fields
localized on each border of the box.
-}
onBorderAttributes : List (Element.Attribute msg)
onBorderAttributes =
    [ Element.centerX, Element.centerY, Element.Background.color Style.white ]


{-| Input field for one of the crop parameters (left, top, right or bottom).
-}
cropField : String -> (String -> msg) -> Field Int IntError -> Element msg
cropField label msgTag field =
    let
        fontColor : Element.Color
        fontColor =
            case field.decodedInput of
                Ok _ ->
                    Style.black

                Err _ ->
                    Style.errorColor
    in
    Element.Input.text
        [ Element.paddingXY 0 4
        , Element.width (Element.px 60)
        , Element.Border.width 0
        , Element.Font.center
        , Element.Font.color fontColor
        ]
        { onChange = msgTag
        , text = field.input
        , placeholder = Nothing
        , label = Element.Input.labelHidden label
        }
