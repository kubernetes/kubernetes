var xmlns_attr =
{
    name: "xmlns", value: "http://foo", prefix: "xmlns",
    local: "", uri : "http://www.w3.org/2000/xmlns/"
};

var attr_attr =
{
    name: "attr", value: "bar", prefix: "",
    local : "attr",  uri : ""
};


require(__dirname).test
  ( { xml :
      "<elm xmlns='http://foo' attr='bar'/>"
    , expect :
      [ [ "opennamespace", { prefix: "", uri: "http://foo" } ]
      , [ "attribute", xmlns_attr ]
      , [ "attribute", attr_attr ]
      , [ "opentag", { name: "elm", prefix: "", local: "elm", uri : "http://foo",
                       ns : { "" : "http://foo" },
                       attributes: { xmlns: xmlns_attr, attr: attr_attr },
                       isSelfClosing: true } ]
      , [ "closetag", "elm" ]
      , [ "closenamespace", { prefix: "", uri: "http://foo"} ]
      ]
    , strict : true
    , opt : {xmlns: true}
    }
  )
