
require(__dirname).test
  ( { xml :
      "<xmlns/>"
    , expect :
      [ [ "opentag", { name: "xmlns", uri: "", prefix: "", local: "xmlns",
                       attributes: {}, ns: {},
                       isSelfClosing: true}
        ],
        ["closetag", "xmlns"]
      ]
    , strict : true
    , opt : { xmlns: true }
    }
  );
