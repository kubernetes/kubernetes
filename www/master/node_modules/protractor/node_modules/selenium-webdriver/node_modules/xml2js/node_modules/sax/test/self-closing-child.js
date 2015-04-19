
require(__dirname).test({
  xml :
  "<root>"+
    "<child>" +
      "<haha />" +
    "</child>" +
    "<monkey>" +
      "=(|)" +
    "</monkey>" +
  "</root>",
  expect : [
    ["opentag", {
     "name": "ROOT",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["opentag", {
     "name": "CHILD",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["opentag", {
     "name": "HAHA",
     "attributes": {},
     "isSelfClosing": true
    }],
    ["closetag", "HAHA"],
    ["closetag", "CHILD"],
    ["opentag", {
     "name": "MONKEY",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["text", "=(|)"],
    ["closetag", "MONKEY"],
    ["closetag", "ROOT"],
    ["end"],
    ["ready"]
  ],
  strict : false,
  opt : {}
});

