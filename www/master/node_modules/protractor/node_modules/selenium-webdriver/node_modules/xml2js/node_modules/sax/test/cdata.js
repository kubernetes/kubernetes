require(__dirname).test({
  xml : "<r><![CDATA[ this is character data  ]]></r>",
  expect : [
    ["opentag", {"name": "R","attributes": {}, "isSelfClosing": false}],
    ["opencdata", undefined],
    ["cdata", " this is character data  "],
    ["closecdata", undefined],
    ["closetag", "R"]
  ]
});
