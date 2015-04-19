// https://github.com/isaacs/sax-js/issues/49
require(__dirname).test
  ( { xml : "<xml><script>hello world</script></xml>"
    , expect :
      [ [ "opentag", { name: "xml", attributes: {}, isSelfClosing: false } ]
      , [ "opentag", { name: "script", attributes: {}, isSelfClosing: false } ]
      , [ "text", "hello world" ]
      , [ "closetag", "script" ]
      , [ "closetag", "xml" ]
      ]
    , strict : false
    , opt : { lowercasetags: true, noscript: true }
    }
  )

require(__dirname).test
  ( { xml : "<xml><script><![CDATA[hello world]]></script></xml>"
    , expect :
      [ [ "opentag", { name: "xml", attributes: {}, isSelfClosing: false } ]
      , [ "opentag", { name: "script", attributes: {}, isSelfClosing: false } ]
      , [ "opencdata", undefined ]
      , [ "cdata", "hello world" ]
      , [ "closecdata", undefined ]
      , [ "closetag", "script" ]
      , [ "closetag", "xml" ]
      ]
    , strict : false
    , opt : { lowercasetags: true, noscript: true }
    }
  )

