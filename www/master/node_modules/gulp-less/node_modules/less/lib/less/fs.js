var fs;
try
{
    fs = require("graceful-fs");
}
catch(e)
{
    fs = require("fs");
}
module.exports = fs;
