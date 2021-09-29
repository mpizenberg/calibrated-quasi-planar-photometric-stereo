module NumberInput exposing
    ( Field, setMinBound
    , IntError(..), intErrorToString, intDefault, setDefaultIntValue, updateInt
    , FloatError(..), floatErrorToString, floatDefault, setDefaultFloatValue, updateFloat
    )

{-| Data for number form inputs

@docs Field, setMinBound, setMaxBound

@docs IntError, intErrorToString, intDefault, setDefaultIntValue, updateInt

@docs FloatError, floatErrorToString, floatDefault, setDefaultFloatValue, updateFloat

-}

import Form.Decoder exposing (Decoder)


type alias Field num err =
    { defaultValue : num
    , min : Maybe num
    , max : Maybe num
    , increase : num -> num
    , decrease : num -> num
    , input : String
    , decodedInput : Result (List err) num
    }


setMinBound : Maybe number -> Field number err -> Field number err
setMinBound newMin field =
    { field | min = newMin }



-- Int


type IntError
    = IntParsingError
    | IntTooSmall { bound : Int, actual : Int }
    | IntTooBig { bound : Int, actual : Int }


intErrorToString : { valueName : String } -> IntError -> String
intErrorToString { valueName } err =
    case err of
        IntParsingError ->
            valueName ++ " is not a valid integer."

        IntTooSmall { bound, actual } ->
            valueName ++ " = " ++ String.fromInt actual ++ " but it should be >= " ++ String.fromInt bound ++ "."

        IntTooBig { bound, actual } ->
            valueName ++ " = " ++ String.fromInt actual ++ " but it should be <= " ++ String.fromInt bound ++ "."


intDefault : Field Int IntError
intDefault =
    { defaultValue = 0
    , min = Nothing
    , max = Nothing
    , increase = \n -> n + 1
    , decrease = \n -> n - 1
    , input = "0"
    , decodedInput = Ok 0
    }


setDefaultIntValue : Int -> Field Int IntError -> Field Int IntError
setDefaultIntValue int field =
    { field | defaultValue = int, input = String.fromInt int, decodedInput = Ok int }


updateInt : String -> Field Int IntError -> Field Int IntError
updateInt input field =
    { field | input = input, decodedInput = Form.Decoder.run (intDecoder field.min field.max) input }


intDecoder : Maybe Int -> Maybe Int -> Decoder String IntError Int
intDecoder maybeMin maybeMax =
    Form.Decoder.int IntParsingError
        |> validateMinInt maybeMin
        |> validateMaxInt maybeMax


validateMinInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMinInt maybeMin decoder =
    case maybeMin of
        Nothing ->
            decoder

        Just minInt ->
            Form.Decoder.assert (minBound IntTooSmall minInt) decoder


validateMaxInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMaxInt maybeMax decoder =
    case maybeMax of
        Nothing ->
            decoder

        Just maxInt ->
            Form.Decoder.assert (maxBound IntTooBig maxInt) decoder



-- Float


type FloatError
    = FloatParsingError
    | FloatTooSmall { bound : Float, actual : Float }
    | FloatTooBig { bound : Float, actual : Float }


floatErrorToString : { valueName : String } -> FloatError -> String
floatErrorToString { valueName } err =
    case err of
        FloatParsingError ->
            valueName ++ " is not a valid number."

        FloatTooSmall { bound, actual } ->
            valueName ++ " = " ++ String.fromFloat actual ++ " but it should be >= " ++ String.fromFloat bound ++ "."

        FloatTooBig { bound, actual } ->
            valueName ++ " = " ++ String.fromFloat actual ++ " but it should be <= " ++ String.fromFloat bound ++ "."


floatDefault : Field Float FloatError
floatDefault =
    { defaultValue = 0.0
    , min = Nothing
    , max = Nothing
    , increase = \n -> n + 0.1
    , decrease = \n -> n - 0.1
    , input = "0.0"
    , decodedInput = Ok 0.0
    }


setDefaultFloatValue : Float -> Field Float FloatError -> Field Float FloatError
setDefaultFloatValue float field =
    { field | defaultValue = float, input = String.fromFloat float, decodedInput = Ok float }


updateFloat : String -> Field Float FloatError -> Field Float FloatError
updateFloat input field =
    { field | input = input, decodedInput = Form.Decoder.run (floatDecoder field.min field.max) input }


floatDecoder : Maybe Float -> Maybe Float -> Decoder String FloatError Float
floatDecoder maybeMin maybeMax =
    Form.Decoder.float FloatParsingError
        |> validateMinFloat maybeMin
        |> validateMaxFloat maybeMax


validateMinFloat : Maybe Float -> Decoder input FloatError Float -> Decoder input FloatError Float
validateMinFloat maybeMin decoder =
    case maybeMin of
        Nothing ->
            decoder

        Just minFloat ->
            Form.Decoder.assert (minBound FloatTooSmall minFloat) decoder


validateMaxFloat : Maybe Float -> Decoder input FloatError Float -> Decoder input FloatError Float
validateMaxFloat maybeMax decoder =
    case maybeMax of
        Nothing ->
            decoder

        Just maxFloat ->
            Form.Decoder.assert (maxBound FloatTooBig maxFloat) decoder



-- Helper


minBound : ({ bound : number, actual : number } -> err) -> number -> Decoder number err ()
minBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual < bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )


maxBound : ({ bound : number, actual : number } -> err) -> number -> Decoder number err ()
maxBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual > bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )
