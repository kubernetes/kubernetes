// default to uppercase
require(__dirname).test
  ( { xml :
      "<span class=\"test\" hello=\"world\"></span>"
    , expect :
      [ [ "attribute", { name: "CLASS", value: "test" } ]
      , [ "attribute", { name: "HELLO", value: "world" } ]
      , [ "opentag", { name: "SPAN",
                       attributes: { CLASS: "test", HELLO: "world" },
                       isSelfClosing: false } ]
      , [ "closetag", "SPAN" ]
      ]
    , strict : false
    , opt : {}
    }
  )

// lowercase option : lowercase tag/attribute names
require(__dirname).test
  ( { xml :
      "<span class=\"test\" hello=\"world\"></span>"
    , expect :
      [ [ "attribute", { name: "class", value: "test" } ]
      , [ "attribute", { name: "hello", value: "world" } ]
      , [ "opentag", { name: "span",
                       attributes: { class: "test", hello: "world" },
                       isSelfClosing: false } ]
      , [ "closetag", "span" ]
      ]
    , strict : false
    , opt : {lowercase:true}
    }
  )

// backward compatibility with old lowercasetags opt
require(__dirname).test
  ( { xml :
      "<span class=\"test\" hello=\"world\"></span>"
    , expect :
      [ [ "attribute", { name: "class", value: "test" } ]
      , [ "attribute", { name: "hello", value: "world" } ]
      , [ "opentag", { name: "span",
                       attributes: { class: "test", hello: "world" },
                       isSelfClosing: false } ]
      , [ "closetag", "span" ]
      ]
    , strict : false
    , opt : {lowercasetags:true}
    }
  )
