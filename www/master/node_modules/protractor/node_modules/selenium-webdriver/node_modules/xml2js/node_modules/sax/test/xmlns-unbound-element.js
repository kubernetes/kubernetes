require(__dirname).test(
  { strict : true
  , opt : { xmlns: true }
  , expect :
    [ [ "error", "Unbound namespace prefix: \"unbound:root\"\nLine: 0\nColumn: 15\nChar: >"]
    , [ "opentag", { name: "unbound:root", uri: "unbound", prefix: "unbound", local: "root"
        , attributes: {}, ns: {}, isSelfClosing: true } ]
    , [ "closetag", "unbound:root" ]
    ]
  }
).write("<unbound:root/>");

require(__dirname).test(
  { strict : true
  , opt : { xmlns: true }
  , expect :
    [ [ "opennamespace", { prefix: "unbound", uri: "someuri" } ]
    , [ "attribute", { name: 'xmlns:unbound', value: 'someuri'
      , prefix: 'xmlns', local: 'unbound'
      , uri: 'http://www.w3.org/2000/xmlns/' } ]
    , [ "opentag", { name: "unbound:root", uri: "someuri", prefix: "unbound", local: "root"
          , attributes: { 'xmlns:unbound': {
              name: 'xmlns:unbound'
            , value: 'someuri'
            , prefix: 'xmlns'
            , local: 'unbound'
            , uri: 'http://www.w3.org/2000/xmlns/' } }
      , ns: { "unbound": "someuri" }, isSelfClosing: true } ]
    , [ "closetag", "unbound:root" ]
    , [ "closenamespace", { prefix: 'unbound', uri: 'someuri' }]
    ]
  }
).write("<unbound:root xmlns:unbound=\"someuri\"/>");
