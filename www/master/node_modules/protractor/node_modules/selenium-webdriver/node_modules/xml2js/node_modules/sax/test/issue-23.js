
require(__dirname).test
  ( { xml :
      "<compileClassesResponse>"+
        "<result>"+
          "<bodyCrc>653724009</bodyCrc>"+
          "<column>-1</column>"+
          "<id>01pG0000002KoSUIA0</id>"+
          "<line>-1</line>"+
          "<name>CalendarController</name>"+
          "<success>true</success>"+
        "</result>"+
      "</compileClassesResponse>"

    , expect :
      [ [ "opentag", { name: "COMPILECLASSESRESPONSE", attributes: {}, isSelfClosing: false } ]
      , [ "opentag", { name : "RESULT", attributes: {}, isSelfClosing: false } ]
      , [ "opentag", { name: "BODYCRC", attributes: {}, isSelfClosing: false } ]
      , [ "text", "653724009" ]
      , [ "closetag", "BODYCRC" ]
      , [ "opentag", { name: "COLUMN", attributes: {}, isSelfClosing: false } ]
      , [ "text", "-1" ]
      , [ "closetag", "COLUMN" ]
      , [ "opentag", { name: "ID", attributes: {}, isSelfClosing: false } ]
      , [ "text", "01pG0000002KoSUIA0" ]
      , [ "closetag", "ID" ]
      , [ "opentag", {name: "LINE", attributes: {}, isSelfClosing: false } ]
      , [ "text", "-1" ]
      , [ "closetag", "LINE" ]
      , [ "opentag", {name: "NAME", attributes: {}, isSelfClosing: false } ]
      , [ "text", "CalendarController" ]
      , [ "closetag", "NAME" ]
      , [ "opentag", {name: "SUCCESS", attributes: {}, isSelfClosing: false } ]
      , [ "text", "true" ]
      , [ "closetag", "SUCCESS" ]
      , [ "closetag", "RESULT" ]
      , [ "closetag", "COMPILECLASSESRESPONSE" ]
      ]
    , strict : false
    , opt : {}
    }
  )

