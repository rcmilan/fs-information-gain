open FSharp.Data

type MyCsv = CsvProvider<"base_mesclada_lead_score_contatos.csv", HasHeaders=true>
let data = MyCsv.GetSample()
let rows = data.Rows |> Seq.toArray
let total = float rows.Length

// Compute entropy for binary variable (p = probability of true)
let entropy p =
    match p with
    | 0.0 | 1.0 -> 0.0
    | _ -> -p * log p / log 2.0 - (1.0 - p) * log (1.0 - p) / log 2.0

// Compute entropy of Comprou
let trueCount = rows |> Array.filter (fun row -> row.Comprou) |> Array.length |> float
let p = trueCount / total
let targetEntropy = entropy p

// Feature extraction function list: name * extractor
let features : (string * (MyCsv.Row -> obj)) list = [
    //"Student_created_at",           fun row -> box row.Student_created_at
    "Acquireurl",                   fun row -> box row.Acquireurl
    "Channel Subscription",         fun row -> box row.``Channel Subscription``
    //"Create Date",                  fun row -> box row.``Create Date``
    //"Email",                        fun row -> box row.Email
    //"First Name",                   fun row -> box row.``First Name``
    "Internalname",                 fun row -> box row.Internalname
    "Is_referred",                  fun row -> box row.Is_referred
    //"Last Name",                    fun row -> box row.``Last Name``
    //"Name",                         fun row -> box row.Name
    "Plan",                         fun row -> box row.Plan
    "Pro Lead Form Objective",      fun row -> box row.``Pro Lead Form Objective;``
    "Pro Lead Form Quando Começar", fun row -> box row.``Pro Lead Form Quando Come�ar``
    //"Record ID",                    fun row -> box row.``Record ID``
    //"Setup_actived_a",              fun row -> box row.Setup_actived_at
    "Sexo",                         fun row -> box row.Sexo
    "Student_age_range",            fun row -> box row.Student_age_range
    //"Student_email",                fun row -> box row.Student_email
    "Student_uf",                   fun row -> box row.Student_uf
    "Utm_medium",                   fun row -> box row.Utm_medium
    "Utm_source",                   fun row -> box row.Utm_source
]

// For each feature, compute conditional entropy
let infoGains =
    features
    |> List.map (fun (name, extractor) ->
        let groups =
            rows
            |> Seq.groupBy extractor
            |> Seq.map (fun (_, group) ->
                let g = group |> Seq.toArray
                let groupSize = float g.Length
                let pTrue = g |> Array.filter (fun r -> r.Comprou) |> Array.length |> float |> fun n -> n / groupSize
                let h = entropy pTrue
                let weighted = (groupSize / total) * h
                weighted
            )
        let conditionalEntropy = groups |> Seq.sum
        let ig = targetEntropy - conditionalEntropy
        (name, ig)
    )
    |> List.sortByDescending snd

// Print results
printfn "Information Gain by column (sorted):"
infoGains
|> List.iter (fun (name, ig) -> printfn "%s: %.5f" name ig)

System.Console.ReadKey() |> ignore
