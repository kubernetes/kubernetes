// https://github.com/isaacs/sax-js/issues/35
require(__dirname).test
  ( { xml : "<xml>&#Xd;&#X0d;\n"+
            "</xml>"

    , expect :
      [ [ "opentag", { name: "xml", attributes: {}, isSelfClosing: false } ]
      , [ "text", "\r\r\n" ]
      , [ "closetag", "xml" ]
      ]
    , strict : true
    , opt : {}
    }
  )

