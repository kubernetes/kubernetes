
require(__dirname).test({
  expect : [
    ["opentag", {"name": "R","attributes": {}, "isSelfClosing": false}],
    ["opencdata", undefined],
    ["cdata", " this is "],
    ["closecdata", undefined],
    ["closetag", "R"]
  ]
})
  .write("<r><![CDATA[ this is ]")
  .write("]>")
  .write("</r>")
  .close();

