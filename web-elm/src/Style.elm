module Style exposing
    ( font, fontMonospace
    , dropColor, runProgressColor, errorColor, warningColor
    , black, white, almostWhite, lightGrey, green
    )

{-| Style of our application

@docs font, fontMonospace

@docs dropColor, runProgressColor, errorColor, warningColor

@docs black, white, almostWhite, lightGrey, green

-}

import Element
import Element.Font as Font


font : Element.Attribute msg
font =
    Font.family
        [ Font.typeface "Open Sans"
        , Font.sansSerif
        ]


fontMonospace : Element.Attribute msg
fontMonospace =
    Font.family
        [ Font.typeface "Inconsolata"
        , Font.monospace
        ]



-- Color


dropColor : Element.Color
dropColor =
    Element.rgb255 50 50 250


runProgressColor : Element.Color
runProgressColor =
    Element.rgb255 211 199 255


errorColor : Element.Color
errorColor =
    Element.rgb255 180 50 50


warningColor : Element.Color
warningColor =
    Element.rgb255 220 120 50


lightGrey : Element.Color
lightGrey =
    Element.rgb255 187 187 187


green : Element.Color
green =
    Element.rgb255 39 203 139


white : Element.Color
white =
    Element.rgb255 255 255 255


almostWhite : Element.Color
almostWhite =
    Element.rgb255 235 235 235


black : Element.Color
black =
    Element.rgb255 0 0 0
