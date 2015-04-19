var xml2js = require('xml2js');
var parser = new xml2js.Parser({
    mergeAttrs: true
});
parser.parseString('<outline xmlUrl="http://www.futurity.org/feed/"/>', function (err, result) {
    console.dir(result);
});
