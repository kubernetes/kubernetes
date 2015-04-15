QUnit.begin = function() {
	console.log("Starting test suite");
	console.log("================================================\n");
};

QUnit.moduleDone = function(opts) {
	if(opts.failed === 0) {
		console.log("\u2714 All tests passed in '"+opts.name+"' module");
	} else {
		console.log("\u2716 "+ opts.failed +" tests failed in '"+opts.name+"' module");
	}
};

QUnit.done = function(opts) {
	console.log("\n================================================");
	console.log("Tests completed in "+opts.runtime+" milliseconds");
	console.log(opts.passed + " tests of "+opts.total+" passed, "+opts.failed+" failed.");
};

module('Basics', {
    setup:function() {
    },
    teardown:function() {
    }
});

test("globals set up", function() {

	ok(window.Modernizr, 'global modernizr object created');

});

test("bind is implemented", function() {

  ok(Function.prototype.bind, 'bind is a member of Function.prototype');

  var a = function(){
      return this.modernizr;
  };
  a = a.bind({modernizr: 'just awsome'});

  equal("just awsome", a(), 'bind works as expected');


  // thank you webkit layoutTests


  var result;

  function F(x, y)
  {
      result = this + " -> x:" + x + ", y:" + y;
  }

  G = F.bind("'a'", "'b'");
  H = G.bind("'Cannot rebind this!'", "'c'");

  G(1,2);
  equal(result, "\'a\' -> x:\'b\', y:1");
  H(1,2);
  equal(result, "\'a\' -> x:\'b\', y:\'c\'");

  var f = new F(1,2);
  equal(result, "[object Object] -> x:1, y:2");
  var g = new G(1,2);
  equal(result, "[object Object] -> x:\'b\', y:1");
  var h = new H(1,2);
  equal(result, "[object Object] -> x:\'b\', y:\'c\'");

  ok(f instanceof F, "f instanceof F");
  ok(g instanceof F, "g instanceof F");
  ok(h instanceof F, "h instanceof F");

  // Bound functions don't have a 'prototype' property.
  ok("prototype" in F, '"prototype" in F');

  // The object passed to bind as 'this' must be callable.
  raises(function(){
    Function.bind.call(undefined);
  });

  // Objects that allow call but not construct can be bound, but should throw if used with new.
  var abcAt = String.prototype.charAt.bind("abc");
  equal(abcAt(1), "b", 'Objects that allow call but not construct can be bound...');

  equal(1, Function.bind.length, 'it exists');


});



test("document.documentElement is valid and correct",1, function() {
	equal(document.documentElement,document.getElementsByTagName('html')[0]);
});


test("no-js class is gone.", function() {

	ok(!/(?:^|\s)no-js(?:^|\s)/.test(document.documentElement.className),
	   'no-js class is gone');

	ok(/(?:^|\s)js(?:^|\s)/.test(document.documentElement.className),
	   'html.js class is present');

	ok(/(?:^|\s)\+no-js(?:\s|$)/.test(document.documentElement.className),
	   'html.+no-js class is still present');

	ok(/(?:^|\s)no-js-(?:\s|$)/.test(document.documentElement.className),
	   'html.no-js- class is still present');

	ok(/(?:^|\s)i-has-no-js(?:\s|$)/.test(document.documentElement.className),
	   'html.i-has-no-js class is still present');

	if (document.querySelector){
	  ok(document.querySelector('html.js') == document.documentElement,
	     "document.querySelector('html.js') matches.");
	}
});

test('html shim worked', function(){
  expect(2);

  // the exact test we use in the script
  var elem = document.getElementsByTagName("section")[0];
  elem.id = "html5section";

  ok( elem.childNodes.length === 1 , 'unknown elements dont collapse');

  elem.style.color = 'red';
  ok( /red|#ff0000/i.test(elem.style.color), 'unknown elements are styleable')

});


module('Modernizr classes and bools', {
    setup:function() {
    },
    teardown:function() {
    }
});


test('html classes are looking good',function(){

  var classes = TEST.trim(document.documentElement.className).split(/\s+/);

  var modprops = Object.keys(Modernizr),
      newprops = modprops;

  // decrement for the properties that are private
  for (var i = -1, len = TEST.privates.length; ++i < len; ){
    var item = TEST.privates[i];
    equal(-1, TEST.inArray(item, classes), 'private Modernizr object '+ item +'should not have matching classes');
    equal(-1, TEST.inArray('no-' + item, classes), 'private Modernizr object no-'+item+' should not have matching classes');
  }

  // decrement for the non-boolean objects
//  for (var i = -1, len = TEST.inputs.length; ++i < len; ){
//    if (Modernizr[TEST.inputs[i]] != undefined) newprops--;
//  }

  // TODO decrement for the extraclasses

  // decrement for deprecated ones.
  $.each( TEST.deprecated, function(key, val){
    newprops.splice(  TEST.inArray(item, newprops), 1);
  });


  //equal(classes,newprops,'equal number of classes and global object props');

  if (classes.length !== newprops){
    //window.console && console.log(classes, newprops);

  }

  for (var i = 0, len = classes.length, aclass; i <len; i++){
    aclass = classes[i];

    // Skip js related classes.
    if (/^(?:js|\+no-js|no-js-|i-has-no-js)$/.test(aclass)) continue;

    if (aclass.indexOf('no-') === 0){
      aclass = aclass.replace('no-','');

      equal(Modernizr[aclass], false,
            aclass + ' is correctly false in the classes and object')

    } else {
      equal(Modernizr[aclass], true,
             aclass + ' is correctly true in the classes and object')
    }
  }


  for (var i = 0, len = classes.length, aclass; i <len; i++){
    equal(classes[i],classes[i].toLowerCase(),'all classes are lowerCase.');
  }

  // Remove fake no-js classes before test.
  var docElClass = document.documentElement.className;
  $.each(['\\+no-js', 'no-js-', 'i-has-no-js'], function(i, fakeClass) {
    docElClass = docElClass.replace(new RegExp('(^|\\s)' + fakeClass + '(\\s|$)', 'g'), '$1$2');
  });
  equal(/[^\s]no-/.test(docElClass), false, 'whitespace between all classes.');


})


test('Modernizr properties are looking good',function(){

  var count  = 0,
      nobool = TEST.API.concat(TEST.inputs)
                       .concat(TEST.audvid)
                       .concat(TEST.privates)
                       .concat(['textarea']); // due to forms-placeholder.js test

  for (var prop in window.Modernizr){
    if (Modernizr.hasOwnProperty(prop)){

      if (TEST.inArray(prop,nobool) >= 0) continue;

      ok(Modernizr[prop] === true || Modernizr[prop] === false,
        'Modernizr.'+prop+' is a straight up boolean');


      equal(prop,prop.toLowerCase(),'all properties are lowerCase.')
    }
  }
})



test('Modernizr.audio and Modernizr.video',function(){

  for (var i = -1, len = TEST.audvid.length; ++i < len;){
    var prop = TEST.audvid[i];

    if (Modernizr[prop].toString() == 'true'){

      ok(Modernizr[prop],                             'Modernizr.'+prop+' is truthy.');
      equal(Modernizr[prop] == true,true,            'Modernizr.'+prop+' is == true')
      equal(typeof Modernizr[prop] === 'object',true,'Moderizr.'+prop+' is truly an object');
      equal(Modernizr[prop] !== true,true,           'Modernizr.'+prop+' is !== true')

    } else {

      equal(Modernizr[prop] != true,true,            'Modernizr.'+prop+' is != true')
    }
  }


});


test('Modernizr results match expected values',function(){

  // i'm bringing over a few tests from inside Modernizr.js
  equal(!!document.createElement('canvas').getContext,Modernizr.canvas,'canvas test consistent');

  equal(!!window.Worker,Modernizr.webworkers,'web workers test consistent')

});



module('Modernizr\'s API methods', {
    setup:function() {
    },
    teardown:function() {
    }
});

test('Modernizr.addTest()',22,function(){

  var docEl = document.documentElement;


  Modernizr.addTest('testtrue',function(){
    return true;
  });

  Modernizr.addTest('testtruthy',function(){
    return 100;
  });

  Modernizr.addTest('testfalse',function(){
    return false;
  });

  Modernizr.addTest('testfalsy',function(){
    return undefined;
  });

  ok(docEl.className.indexOf(' testtrue') >= 0,'positive class added');
  equal(Modernizr.testtrue,true,'positive prop added');

  ok(docEl.className.indexOf(' testtruthy') >= 0,'positive class added');
  equal(Modernizr.testtruthy,100,'truthy value is not casted to straight boolean');

  ok(docEl.className.indexOf(' no-testfalse') >= 0,'negative class added');
  equal(Modernizr.testfalse,false,'negative prop added');

  ok(docEl.className.indexOf(' no-testfalsy') >= 0,'negative class added');
  equal(Modernizr.testfalsy,undefined,'falsy value is not casted to straight boolean');



  Modernizr.addTest('testcamelCase',function(){
     return true;
   });

  ok(docEl.className.indexOf(' testcamelCase') === -1,
     'camelCase test name toLowerCase()\'d');


  // okay new signature for this API! woo

  Modernizr.addTest('testboolfalse', false);

  ok(~docEl.className.indexOf(' no-testboolfalse'), 'Modernizr.addTest(feature, bool): negative class added');
  equal(Modernizr.testboolfalse, false, 'Modernizr.addTest(feature, bool): negative prop added');



  Modernizr.addTest('testbooltrue', true);

  ok(~docEl.className.indexOf(' testbooltrue'), 'Modernizr.addTest(feature, bool): positive class added');
  equal(Modernizr.testbooltrue, true, 'Modernizr.addTest(feature, bool): positive prop added');



  Modernizr.addTest({'testobjboolfalse': false,
                     'testobjbooltrue' : true   });

  ok(~docEl.className.indexOf(' no-testobjboolfalse'), 'Modernizr.addTest({feature: bool}): negative class added');
  equal(Modernizr.testobjboolfalse, false, 'Modernizr.addTest({feature: bool}): negative prop added');

  ok(~docEl.className.indexOf(' testobjbooltrue'), 'Modernizr.addTest({feature: bool}): positive class added');
  equal(Modernizr.testobjbooltrue, true, 'Modernizr.addTest({feature: bool}): positive prop added');




  Modernizr.addTest({'testobjfnfalse': function(){ return false },
                     'testobjfntrue' : function(){ return true }   });


  ok(~docEl.className.indexOf(' no-testobjfnfalse'), 'Modernizr.addTest({feature: bool}): negative class added');
  equal(Modernizr.testobjfnfalse, false, 'Modernizr.addTest({feature: bool}): negative prop added');

  ok(~docEl.className.indexOf(' testobjfntrue'), 'Modernizr.addTest({feature: bool}): positive class added');
  equal(Modernizr.testobjfntrue, true, 'Modernizr.addTest({feature: bool}): positive prop added');


  Modernizr
    .addTest('testchainone', true)
    .addTest({ testchaintwo: true })
    .addTest('testchainthree', function(){ return true; });

  ok( Modernizr.testchainone == Modernizr.testchaintwo == Modernizr.testchainthree, 'addTest is chainable');


}); // eo addTest





test('Modernizr.mq: media query testing',function(){

  var $html = $('html');
  $.mobile = {};

  // from jquery mobile

  $.mobile.media = (function() {
  	// TODO: use window.matchMedia once at least one UA implements it
  	var cache = {},
  		testDiv = $( "<div id='jquery-mediatest'>" ),
  		fakeBody = $( "<body>" ).append( testDiv );

  	return function( query ) {
  		if ( !( query in cache ) ) {
  			var styleBlock = document.createElement('style'),
          		cssrule = "@media " + query + " { #jquery-mediatest { position:absolute; } }";
  	        //must set type for IE!
  	        styleBlock.type = "text/css";
  	        if (styleBlock.styleSheet){
  	          styleBlock.styleSheet.cssText = cssrule;
  	        }
  	        else {
  	          styleBlock.appendChild(document.createTextNode(cssrule));
  	        }

  			$html.prepend( fakeBody ).prepend( styleBlock );
  			cache[ query ] = testDiv.css( "position" ) === "absolute";
  			fakeBody.add( styleBlock ).remove();
  		}
  		return cache[ query ];
  	};
  })();


  ok(Modernizr.mq,'Modernizr.mq() doesn\' freak out.');

  equal($.mobile.media('only screen'), Modernizr.mq('only screen'),'screen media query matches jQuery mobile\'s result');

  equal(Modernizr.mq('only all'), Modernizr.mq('only all'), 'Cache hit matches');


});




test('Modernizr.hasEvent()',function(){

  ok(typeof Modernizr.hasEvent == 'function','Modernizr.hasEvent() is a function');


  equal(Modernizr.hasEvent('click'), true,'click event is supported');

  equal(Modernizr.hasEvent('modernizrcustomevent'), false,'random event is definitely not supported');

  /* works fine in webkit but not gecko
  equal(  Modernizr.hasEvent('resize', window),
          !Modernizr.hasEvent('resize', document.createElement('div')),
          'Resize is supported in window but not a div, typically...');
  */

});





test('Modernizr.testStyles()',function(){

  equal(typeof Modernizr.testStyles, 'function','Modernizr.testStyles() is a function');

  var style = '#modernizr{ width: 9px; height: 4px; font-size: 0; color: papayawhip; }';

  Modernizr.testStyles(style, function(elem, rule){
      equal(style, rule, 'rule passsed back matches what i gave it.')
      equal(elem.offsetWidth, 9, 'width was set through the style');
      equal(elem.offsetHeight, 4, 'height was set through the style');
      equal(elem.id, 'modernizr', 'element is indeed the modernizr element');
  });

});


test('Modernizr._[properties]',function(){

  equal(6, Modernizr._prefixes.length, 'Modernizr._prefixes has 6 items');

  equal(4, Modernizr._domPrefixes.length, 'Modernizr.domPrefixes has 4 items');

});

test('Modernizr.testProp()',function(){

  equal(true, Modernizr.testProp('margin'), 'Everyone supports margin');

  equal(false, Modernizr.testProp('happiness'), 'Nobody supports the happiness style. :(');
  equal(true, Modernizr.testProp('fontSize'), 'Everyone supports fontSize');
  equal(false, Modernizr.testProp('font-size'), 'Nobody supports font-size');

  equal('pointerEvents' in  document.createElement('div').style,
         Modernizr.testProp('pointerEvents'),
         'results for `pointer-events` are consistent with a homegrown feature test');

});



test('Modernizr.testAllProps()',function(){

  equal(true, Modernizr.testAllProps('margin'), 'Everyone supports margin');

  equal(false, Modernizr.testAllProps('happiness'), 'Nobody supports the happiness style. :(');
  equal(true, Modernizr.testAllProps('fontSize'), 'Everyone supports fontSize');
  equal(false, Modernizr.testAllProps('font-size'), 'Nobody supports font-size');

  equal(Modernizr.csstransitions, Modernizr.testAllProps('transition'), 'Modernizr result matches API result: csstransitions');

  equal(Modernizr.csscolumns, Modernizr.testAllProps('columnCount'), 'Modernizr result matches API result: csscolumns')

});






test('Modernizr.prefixed() - css and DOM resolving', function(){
  // https://gist.github.com/523692

  function gimmePrefix(prop, obj){
    var prefixes = ['Moz','Khtml','Webkit','O','ms'],
        domPrefixes = ['moz','khtml','webkit','o','ms'],
        elem     = document.createElement('div'),
        upper    = prop.charAt(0).toUpperCase() + prop.slice(1);

    if(!obj) {
      if (prop in elem.style)
        return prop;

      for (var len = prefixes.length; len--; ){
        if ((prefixes[len] + upper)  in elem.style)
          return (prefixes[len] + upper);
      }
    } else {
      if (prop in obj)
        return prop;

      for (var len = domPrefixes.length; len--; ){
        if ((domPrefixes[len] + upper)  in obj)
          return (domPrefixes[len] + upper);
      }
    }


    return false;
  }

  var propArr = ['transition', 'backgroundSize', 'boxSizing', 'borderImage',
                 'borderRadius', 'boxShadow', 'columnCount'];

  var domPropArr = [{ 'prop': 'requestAnimationFrame',  'obj': window },
                    { 'prop': 'querySelectorAll',       'obj': document },
                    { 'prop': 'matchesSelector',        'obj': document.createElement('div') }];

  for (var i = -1, len = propArr.length; ++i < len; ){
    var prop = propArr[i];
    equal(Modernizr.prefixed(prop), gimmePrefix(prop), 'results for ' + prop + ' match the homebaked prefix finder');
  }

  for (var i = -1, len = domPropArr.length; ++i < len; ){
    var prop = domPropArr[i];
    ok(!!~Modernizr.prefixed(prop.prop, prop.obj, false).toString().indexOf(gimmePrefix(prop.prop, prop.obj)), 'results for ' + prop.prop + ' match the homebaked prefix finder');
  }




});


// FIXME: so a few of these are whitelisting for webkit. i'd like to improve that.
test('Modernizr.prefixed autobind', function(){

  var rAFName;

  // quick sniff to find the local rAF prefixed name.
  var vendors = ['ms', 'moz', 'webkit', 'o'];
  for(var x = 0; x < vendors.length && !rAFName; ++x) {
    rAFName = window[vendors[x]+'RequestAnimationFrame'] && vendors[x]+'RequestAnimationFrame';
  }

  if (rAFName){
    // rAF returns a function
    equal(
      'function',
      typeof Modernizr.prefixed('requestAnimationFrame', window),
      "Modernizr.prefixed('requestAnimationFrame', window) returns a function")

    // unless we false it to a string
    equal(
      rAFName,
      Modernizr.prefixed('requestAnimationFrame', window, false),
      "Modernizr.prefixed('requestAnimationFrame', window, false) returns a string (the prop name)")

  }

  if (document.body.webkitMatchesSelector || document.body.mozMatchesSelector){

    var fn = Modernizr.prefixed('matchesSelector', HTMLElement.prototype, document.body);

    //returns function
    equal(
      'function',
      typeof fn,
      "Modernizr.prefixed('matchesSelector', HTMLElement.prototype, document.body) returns a function");

      // fn scoping
    equal(
      true,
      fn('body'),
      "Modernizr.prefixed('matchesSelector', HTMLElement.prototype, document.body) is scoped to the body")

  }

  // Webkit only: are there other objects that are prefixed?
  if (window.webkitNotifications){
    // should be an object.

    equal(
      'object',
      typeof Modernizr.prefixed('Notifications', window),
      "Modernizr.prefixed('Notifications') returns an object");

  }

  // Webkit only:
  if (typeof document.webkitIsFullScreen !== 'undefined'){
    // boolean

    equal(
      'boolean',
      typeof Modernizr.prefixed('isFullScreen', document),
      "Modernizr.prefixed('isFullScreen') returns a boolean");
  }



  // Moz only:
  if (typeof document.mozFullScreen !== 'undefined'){
    // boolean

    equal(
      'boolean',
      typeof Modernizr.prefixed('fullScreen', document),
      "Modernizr.prefixed('fullScreen') returns a boolean");
  }


  // Webkit-only.. takes advantage of Webkit's mixed case of prefixes
  if (document.body.style.WebkitAnimation){
    // string

    equal(
      'string',
      typeof Modernizr.prefixed('animation', document.body.style),
      "Modernizr.prefixed('animation', document.body.style) returns value of that, as a string");

    equal(
      animationStyle.toLowerCase(),
      Modernizr.prefixed('animation', document.body.style, false).toLowerCase(),
      "Modernizr.prefixed('animation', document.body.style, false) returns the (case-normalized) name of the property: webkitanimation");

  }

  equal(
    false,
    Modernizr.prefixed('doSomethingAmazing$#$', window),
    "Modernizr.prefixed('doSomethingAmazing$#$', window) : Gobbledygook with prefixed(str,obj) returns false");

  equal(
    false,
    Modernizr.prefixed('doSomethingAmazing$#$', window, document.body),
    "Modernizr.prefixed('doSomethingAmazing$#$', window) : Gobbledygook with prefixed(str,obj, scope) returns false");


  equal(
    false,
    Modernizr.prefixed('doSomethingAmazing$#$', window, false),
    "Modernizr.prefixed('doSomethingAmazing$#$', window) : Gobbledygook with prefixed(str,obj, false) returns false");


});





