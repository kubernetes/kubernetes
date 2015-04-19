require(__dirname).test({
  xml : "<html><head><script>'<div>foo</div></'</script></head></html>",
  expect : [
    ["opentag", {"name": "HTML","attributes": {}, isSelfClosing: false}],
    ["opentag", {"name": "HEAD","attributes": {}, isSelfClosing: false}],
    ["opentag", {"name": "SCRIPT","attributes": {}, isSelfClosing: false}],
    ["script", "'<div>foo</div></'"],
    ["closetag", "SCRIPT"],
    ["closetag", "HEAD"],
    ["closetag", "HTML"]
  ]
});
