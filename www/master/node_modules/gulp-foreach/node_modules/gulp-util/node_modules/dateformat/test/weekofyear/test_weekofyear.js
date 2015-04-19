var dateFormat = require('../lib/dateformat.js');

var val = process.argv[2] || new Date();
console.log(dateFormat(val, 'W'));
