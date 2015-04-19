// Support for very old versions of node where the module was called "sys".  At some point, we should abandon this.

var util;

try {
    util = require("util");
} catch (err) {
    util = require("sys");
}

module.exports = util;
