var p = require("webpage").create();

p.onConsoleMessage = function(msg) { console.log(msg); };

// Calls to "callPhantom" within the page 'p' arrive here
p.onCallback = function(msg) {
    console.log("Received by the 'phantom' main context: "+msg);
    return "Hello there, I'm coming to you from the 'phantom' context instead";
};

p.evaluate(function() {
    // Return-value of the "onCallback" handler arrive here
    var callbackResponse = window.callPhantom("Hello, I'm coming to you from the 'page' context");
    console.log("Received by the 'page' context: "+callbackResponse);
});

phantom.exit();
