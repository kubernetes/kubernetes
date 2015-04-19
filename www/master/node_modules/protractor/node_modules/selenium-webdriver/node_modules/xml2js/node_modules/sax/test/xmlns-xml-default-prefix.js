require(__dirname).test(
  { xml : "<xml:root/>"
  , expect :
    [
      [ "opentag"
      , { name: "xml:root"
        , uri: "http://www.w3.org/XML/1998/namespace"
        , prefix: "xml"
        , local: "root"
        , attributes: {}
        , ns: {}
        , isSelfClosing: true
        }
      ]
    , ["closetag", "xml:root"]
    ]
  , strict : true
  , opt : { xmlns: true }
  }
)

