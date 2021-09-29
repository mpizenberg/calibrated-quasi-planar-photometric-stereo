module ReviewConfig exposing (config)

{-| Do not rename the ReviewConfig module or the config function, because
`elm-review` will look for these.

To add packages that contain rules, add them to this review project using

    `elm install author/packagename`

when inside the directory containing this file.

-}

import NoDebug.Log
import NoDebug.TodoOrToString
import NoExposingEverything
import NoImportingEverything
import NoMissingSubscriptionsCall
import NoMissingTypeAnnotation
import NoMissingTypeAnnotationInLetIn
import NoMissingTypeExpose
import NoRecursiveUpdate
import NoUnoptimizedRecursion
import NoUnused.CustomTypeConstructorArgs
import NoUnused.Dependencies
import NoUnused.Exports
import NoUnused.Modules
import NoUnused.Parameters
import NoUnused.Patterns
import NoUnused.Variables
import NoUselessSubscriptions
import Review.Rule exposing (Rule)
import Simplify


config : List Rule
config =
    -- unused things
    [ NoUnused.CustomTypeConstructorArgs.rule
    , NoUnused.Dependencies.rule
    , NoUnused.Exports.rule
    , NoUnused.Modules.rule
    , NoUnused.Parameters.rule
    , NoUnused.Patterns.rule
    , NoUnused.Variables.rule

    -- misusing concerning the elm architecture
    , NoMissingSubscriptionsCall.rule
    , NoRecursiveUpdate.rule
    , NoUselessSubscriptions.rule

    -- misc
    , NoExposingEverything.rule
    , NoImportingEverything.rule [ "Svg.Attributes" ]
    , NoMissingTypeAnnotation.rule
    , NoMissingTypeAnnotationInLetIn.rule
    , NoMissingTypeExpose.rule
        |> Review.Rule.ignoreErrorsForFiles [ "src/Main.elm" ]

    -- no debug in code to permit --optimize for make
    , NoDebug.Log.rule
    , NoDebug.TodoOrToString.rule
    , Simplify.rule Simplify.defaults

    -- force tail-call recursions
    , NoUnoptimizedRecursion.rule
        (NoUnoptimizedRecursion.optOutWithComment "IGNORE TAIL-OPTI")
    ]
