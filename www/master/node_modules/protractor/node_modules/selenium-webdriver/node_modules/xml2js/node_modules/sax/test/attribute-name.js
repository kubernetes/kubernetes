require(__dirname).test(
  { xml: "<root length='12345'></root>"
  , expect: [
    ["attribute", {
      name: "length"
    , value: "12345"
    , prefix: ""
    , local: "length"
    , uri: ""
    }]
  , ["opentag", {
      name: "root"
    , prefix: ""
    , local: "root"
    , uri: ""
    , attributes: {
      length: {
        name: "length"
      , value: "12345"
      , prefix: ""
      , local: "length"
      , uri: ""
      }
    }
    , ns: {}
    , isSelfClosing: false
    }]
  , ["closetag", "root"]
  ]
  , strict: true
  , opt: { xmlns: true }
  }
)
