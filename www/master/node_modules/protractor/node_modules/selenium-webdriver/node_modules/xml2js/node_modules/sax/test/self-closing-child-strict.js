
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
     "name": "root",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["opentag", {
     "name": "child",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["opentag", {
     "name": "haha",
     "attributes": {},
     "isSelfClosing": true
    }],
    ["closetag", "haha"],
    ["closetag", "child"],
    ["opentag", {
     "name": "monkey",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["text", "=(|)"],
    ["closetag", "monkey"],
    ["closetag", "root"],
    ["end"],
    ["ready"]
  ],
  strict : true,
  opt : {}
});

