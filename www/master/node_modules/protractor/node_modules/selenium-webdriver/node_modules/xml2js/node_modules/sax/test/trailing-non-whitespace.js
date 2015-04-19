
require(__dirname).test({
  xml : "<span>Welcome,</span> to monkey land",
  expect : [
    ["opentag", {
     "name": "SPAN",
     "attributes": {},
     isSelfClosing: false
    }],
    ["text", "Welcome,"],
    ["closetag", "SPAN"],
    ["text", " to monkey land"],
    ["end"],
    ["ready"]
  ],
  strict : false,
  opt : {}
});
