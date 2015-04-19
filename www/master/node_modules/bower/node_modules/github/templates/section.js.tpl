/**
 *  mixin <%sectionName%>
 *
 *  Copyright 2012 Cloud9 IDE, Inc.
 *
 *  This product includes software developed by
 *  Cloud9 IDE, Inc (http://c9.io).
 *
 *  Author: Mike de Boer <info@mikedeboer.nl>
 **/

"use strict";

var error = require("./../../error");
var Util = require("./../../util");

var <%sectionName%> = module.exports = {
    <%sectionName%>: {}
};

(function() {
<%sectionBody%>
}).call(<%sectionName%>.<%sectionName%>);
