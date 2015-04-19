/**
 *  mixin statuses
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

var statuses = module.exports = {
    statuses: {}
};

(function() {
    /** section: github
     *  statuses#get(msg, callback) -> null
     *      - msg (Object): Object that contains the parameters and their values to be sent to the server.
     *      - callback (Function): function to call when the request is finished with an error as first argument and result data as second argument.
     *
     *  ##### Params on the `msg` object:
     *
     *  - headers (Object): Optional. Key/ value pair of request headers to pass along with the HTTP request. Valid headers are: 'If-Modified-Since', 'If-None-Match', 'Cookie', 'User-Agent', 'Accept', 'X-GitHub-OTP'.
     *  - user (String): Required. 
     *  - repo (String): Required. 
     *  - sha (String): Required. 
     **/
    this.get = function(msg, block, callback) {
        var self = this;
        this.client.httpSend(msg, block, function(err, res) {
            if (err)
                return self.sendError(err, null, msg, callback);

            var ret;
            try {
                ret = res.data && JSON.parse(res.data);
            }
            catch (ex) {
                if (callback)
                    callback(new error.InternalServerError(ex.message), res);
                return;
            }

            if (!ret)
                ret = {};
            if (!ret.meta)
                ret.meta = {};
            ["x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "x-oauth-scopes", "link", "location", "last-modified", "etag", "status"].forEach(function(header) {
                if (res.headers[header])
                    ret.meta[header] = res.headers[header];
            });

            if (callback)
                callback(null, ret);
        });
    };

    /** section: github
     *  statuses#getCombined(msg, callback) -> null
     *      - msg (Object): Object that contains the parameters and their values to be sent to the server.
     *      - callback (Function): function to call when the request is finished with an error as first argument and result data as second argument.
     *
     *  ##### Params on the `msg` object:
     *
     *  - headers (Object): Optional. Key/ value pair of request headers to pass along with the HTTP request. Valid headers are: 'If-Modified-Since', 'If-None-Match', 'Cookie', 'User-Agent', 'Accept', 'X-GitHub-OTP'.
     *  - user (String): Required. 
     *  - repo (String): Required. 
     *  - sha (String): Required. 
     **/
    this.getCombined = function(msg, block, callback) {
        var self = this;
        this.client.httpSend(msg, block, function(err, res) {
            if (err)
                return self.sendError(err, null, msg, callback);

            var ret;
            try {
                ret = res.data && JSON.parse(res.data);
            }
            catch (ex) {
                if (callback)
                    callback(new error.InternalServerError(ex.message), res);
                return;
            }

            if (!ret)
                ret = {};
            if (!ret.meta)
                ret.meta = {};
            ["x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "x-oauth-scopes", "link", "location", "last-modified", "etag", "status"].forEach(function(header) {
                if (res.headers[header])
                    ret.meta[header] = res.headers[header];
            });

            if (callback)
                callback(null, ret);
        });
    };

    /** section: github
     *  statuses#create(msg, callback) -> null
     *      - msg (Object): Object that contains the parameters and their values to be sent to the server.
     *      - callback (Function): function to call when the request is finished with an error as first argument and result data as second argument.
     *
     *  ##### Params on the `msg` object:
     *
     *  - headers (Object): Optional. Key/ value pair of request headers to pass along with the HTTP request. Valid headers are: 'If-Modified-Since', 'If-None-Match', 'Cookie', 'User-Agent', 'Accept', 'X-GitHub-OTP'.
     *  - user (String): Required. 
     *  - repo (String): Required. 
     *  - sha (String): Required. 
     *  - state (String): Required. State of the status - can be one of pending, success, error, or failure. Validation rule: ` ^(pending|success|error|failure)$ `.
     *  - target_url (String): Optional. Target url to associate with this status. This URL will be linked from the GitHub UI to allow users to easily see the ‘source’ of the Status.
     *  - description (String): Optional. Short description of the status.
     *  - context (String): Optional. A string label to differentiate this status from the status of other systems.
     **/
    this.create = function(msg, block, callback) {
        var self = this;
        this.client.httpSend(msg, block, function(err, res) {
            if (err)
                return self.sendError(err, null, msg, callback);

            var ret;
            try {
                ret = res.data && JSON.parse(res.data);
            }
            catch (ex) {
                if (callback)
                    callback(new error.InternalServerError(ex.message), res);
                return;
            }

            if (!ret)
                ret = {};
            if (!ret.meta)
                ret.meta = {};
            ["x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "x-oauth-scopes", "link", "location", "last-modified", "etag", "status"].forEach(function(header) {
                if (res.headers[header])
                    ret.meta[header] = res.headers[header];
            });

            if (callback)
                callback(null, ret);
        });
    };

}).call(statuses.statuses);
