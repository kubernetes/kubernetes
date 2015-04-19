"use strict";

// long form
angular.module("MyMod").controller("MyCtrl", function($a, $b) {
});

// w/ dependencies
angular.module("MyMod", ["OtherMod"]).controller("MyCtrl", function($c, $d, $e, $f, $g, $h, $i) {
});

// simple
myMod.service("$a", function($b) {
});
myMod.controller("foo", function($a, $b) {
});
myMod.service("foo", function($c, $d) {
});
myMod.factory("foo", function($e, $f) {
});
myMod.directive("foo", function($g, $h) {
});
myMod.filter("foo", function($i, $a) {
});
myMod.animation("foo", function($b, $c) {
});
myMod.invoke("foo", function($d, $e) {
});

// implicit config function
angular.module("MyMod", ["OtherMod"], function($interpolateProvider) {}).controller("foo", function($f) {});

// object property
var myObj = {};
myObj.myMod = angular.module("MyMod");
myObj.myMod.controller("foo", function($g, $h) { a });

// run, config don't take names
myMod.run(function($i, $a) {
});
angular.module("MyMod").run(function($b) {
});
myMod.config(function($c, $d) {
});
angular.module("MyMod").config(function() {
});

// directive return object
myMod.directive("foo", function($e) {
    return {
        controller: function($f, $g) {
            bar;
        }
    }
});
myMod.directive("foo", function($h) {
    return {
        controller: function() {
            bar;
        }
    }
});

// provider, provider $get
myMod.provider("foo", function($i) {
    this.$get = function($a, $b) {
        bar;
    };
    self.$get = function($c) {};
    that.$get = function($d) {};
    ignore.$get = function($e) {};
});
myMod.provider("foo", function() {
    this.$get = function($f) {
        bar;
    };
});
myMod.provider("foo", function() {
    return {
        $get: function($g, $h) {
            bar;
        }};
});
myMod.provider("foo", {
    $get: function($i) {
        bar;
    }
});
