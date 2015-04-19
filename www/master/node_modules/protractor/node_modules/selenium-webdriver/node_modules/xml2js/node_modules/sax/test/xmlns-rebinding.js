
require(__dirname).test
  ( { xml :
      "<root xmlns:x='x1' xmlns:y='y1' x:a='x1' y:a='y1'>"+
        "<rebind xmlns:x='x2'>"+
          "<check x:a='x2' y:a='y1'/>"+
        "</rebind>"+
        "<check x:a='x1' y:a='y1'/>"+
      "</root>"

    , expect :
      [ [ "opennamespace", { prefix: "x", uri: "x1" } ]
      , [ "opennamespace", { prefix: "y", uri: "y1" } ]
      , [ "attribute", { name: "xmlns:x", value: "x1", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "x" } ]
      , [ "attribute", { name: "xmlns:y", value: "y1", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "y" } ]
      , [ "attribute", { name: "x:a", value: "x1", uri: "x1", prefix: "x", local: "a" } ]
      , [ "attribute", { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } ]
      , [ "opentag", { name: "root", uri: "", prefix: "", local: "root",
            attributes: { "xmlns:x": { name: "xmlns:x", value: "x1", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "x" }
                        , "xmlns:y": { name: "xmlns:y", value: "y1", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "y" }
                        , "x:a": { name: "x:a", value: "x1", uri: "x1", prefix: "x", local: "a" }
                        , "y:a": { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } },
            ns: { x: 'x1', y: 'y1' },
            isSelfClosing: false } ]

      , [ "opennamespace", { prefix: "x", uri: "x2" } ]
      , [ "attribute", { name: "xmlns:x", value: "x2", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "x" } ]
      , [ "opentag", { name: "rebind", uri: "", prefix: "", local: "rebind",
            attributes: { "xmlns:x": { name: "xmlns:x", value: "x2", uri: "http://www.w3.org/2000/xmlns/", prefix: "xmlns", local: "x" } },
            ns: { x: 'x2' },
            isSelfClosing: false } ]

      , [ "attribute", { name: "x:a", value: "x2", uri: "x2", prefix: "x", local: "a" } ]
      , [ "attribute", { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } ]
      , [ "opentag", { name: "check", uri: "", prefix: "", local: "check",
            attributes: { "x:a": { name: "x:a", value: "x2", uri: "x2", prefix: "x", local: "a" }
                        , "y:a": { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } },
            ns: { x: 'x2' },
            isSelfClosing: true } ]

      , [ "closetag", "check" ]

      , [ "closetag", "rebind" ]
      , [ "closenamespace", { prefix: "x", uri: "x2" } ]

      , [ "attribute", { name: "x:a", value: "x1", uri: "x1", prefix: "x", local: "a" } ]
      , [ "attribute", { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } ]
      , [ "opentag", { name: "check", uri: "", prefix: "", local: "check",
            attributes: { "x:a": { name: "x:a", value: "x1", uri: "x1", prefix: "x", local: "a" }
                        , "y:a": { name: "y:a", value: "y1", uri: "y1", prefix: "y", local: "a" } },
            ns: { x: 'x1', y: 'y1' },
            isSelfClosing: true } ]
      , [ "closetag", "check" ]

      , [ "closetag", "root" ]
      , [ "closenamespace", { prefix: "x", uri: "x1" } ]
      , [ "closenamespace", { prefix: "y", uri: "y1" } ]
      ]
    , strict : true
    , opt : { xmlns: true }
    }
  )

