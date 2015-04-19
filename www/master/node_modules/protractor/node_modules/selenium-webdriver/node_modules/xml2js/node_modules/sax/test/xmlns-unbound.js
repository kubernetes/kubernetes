
require(__dirname).test(
  { strict : true
  , opt : { xmlns: true }
  , expect :
    [ ["error", "Unbound namespace prefix: \"unbound\"\nLine: 0\nColumn: 28\nChar: >"]

    , [ "attribute", { name: "unbound:attr", value: "value", uri: "unbound", prefix: "unbound", local: "attr" } ]
    , [ "opentag", { name: "root", uri: "", prefix: "", local: "root",
          attributes: { "unbound:attr": { name: "unbound:attr", value: "value", uri: "unbound", prefix: "unbound", local: "attr" } },
          ns: {}, isSelfClosing: true } ]
    , [ "closetag", "root" ]
    ]
  }
).write("<root unbound:attr='value'/>")
