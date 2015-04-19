
require(__dirname).test({
  xml :
  "<root attrib>",
  expect : [
    ["attribute", {name:"ATTRIB", value:"attrib"}],
    ["opentag", {name:"ROOT", attributes:{"ATTRIB":"attrib"}, isSelfClosing: false}]
  ],
  opt : { trim : true }
});
