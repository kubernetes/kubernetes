// https://github.com/isaacs/sax-js/issues/47
require(__dirname).test
  ( { xml : '<a href="query.svc?x=1&y=2&z=3"/>'
    , expect : [
        [ "attribute", { name:'HREF', value:"query.svc?x=1&y=2&z=3"} ],
        [ "opentag", { name: "A", attributes: { HREF:"query.svc?x=1&y=2&z=3"}, isSelfClosing: true } ],
        [ "closetag", "A" ]
      ]
    , opt : {}
    }
  )

