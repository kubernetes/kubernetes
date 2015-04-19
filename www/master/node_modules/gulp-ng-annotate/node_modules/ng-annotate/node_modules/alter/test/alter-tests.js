"use strict";

var test = require("tap").test;
var alter = require("../");

test("simple", function(t) {
    t.equal(alter("0123456789", [
        {start: 1, end: 3, str: "first"},
        {start: 5, end: 9, str: "second"},
    ]), "0first34second9");
    t.end();
});

test("not-sorted-order", function(t) {
    t.equal(alter("0123456789", [
        {start: 5, end: 9, str: "second"},
        {start: 1, end: 3, str: "first"},
    ]), "0first34second9");
    t.end();
});

test("insert", function(t) {
    t.equal(alter("0123456789", [
        {start: 5, end: 5, str: "xyz"},
    ]), "01234xyz56789");
    t.end();
});

test("delete", function(t) {
    t.equal(alter("0123456789", [
        {start: 5, end: 6, str: ""},
    ]), "012346789");
    t.end();
});

test("nop1", function(t) {
    t.equal(alter("0123456789", [
    ]), "0123456789");
    t.end();
});

test("nop2", function(t) {
    t.equal(alter("0123456789", [
        {start: 5, end: 5, str: ""},
    ]), "0123456789");
    t.end();
});

test("orderedinsert-stable", function(t) {
    t.equal(alter("0123456789", [
        {start: 5, end: 5, str: "a"},
        {start: 5, end: 5, str: "b"},
        {start: 5, end: 5, str: "c"},
        {start: 5, end: 6, str: "d"},
    ]), "01234abcd6789");
    t.end();
});
