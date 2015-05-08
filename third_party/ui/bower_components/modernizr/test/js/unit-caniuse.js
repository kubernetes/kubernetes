

var myscript = document.createElement('script'),
    ref = document.getElementsByTagName('script')[0];

myscript.src = 'http://caniuse.com/jsonp.php?callback=caniusecb';

setTimeout(function(){
  ref.parentNode.insertBefore(myscript, ref);
}, 100);

// mapping Modernizr terms over to caniuse terms
var map = {
  audio : 'audio',
  borderimage : 'border-image',
  borderradius : 'border-radius',
  canvas : 'canvas',
  canvastext : 'canvas-text',
  cssanimations : 'css-animation',
  boxshadow : 'css-boxshadow',
  cssgradients : 'css-gradients',
  opacity : 'css-opacity',
  cssreflections : 'css-reflections',
  textshadow : 'css-textshadow',
  csstransitions : 'css-transitions',
  hsla : 'css3-colors',
  rgba : 'css3-colors',
  draganddrop : 'dragndrop',
  flexbox : 'flexbox',
  fontface : 'fontface',
  geolocation : 'geolocation',
  hashchange : 'hashchange',
  history : 'history',
  indexeddb : 'indexeddb',
  multiplebgs : 'multibackgrounds',
  csscolumns : 'multicolumn',
  localstorage : 'namevalue-storage',
  applicationcache : 'offline-apps',
  websqldatabase : 'sql-storage',
  svg : 'svg',
  touch : 'touch',
  csstransforms : 'transforms2d',
  csstransforms3d : 'transforms3d',
  video: 'video',
  webgl: 'webgl',
  websockets : 'websockets',
  webworkers : 'webworkers',
  postmessage : 'x-doc-messaging'
};

window.caniusecb = function(scriptdata) {

  window.doo = scriptdata;

  // quit if JSONSelect didn't make it.
  if (!window.JSONSelect) return;

  var testdata     = scriptdata.data,

      // parse the current UA with uaparser
      ua           = uaparse(navigator.userAgent),

      // match the UA from uaparser into the browser used by caniuse
      browserKey   = JSONSelect.match('.agents .browser', scriptdata).indexOf(ua.family),
      currBrowser  = Object.keys(scriptdata.agents)[browserKey];

  // So Phantom doesn't kill the caniuse.com matching exit out as it's useless anyway within PhantomJS
  if(navigator.userAgent.indexOf("PhantomJS") != -1) {
    return;
  }

  // translate 'y' 'n' or 'a' into a boolean that Modernizr uses
  function bool(ciuresult){
    if (ciuresult == 'y' || ciuresult == 'a') return true;
    // 'p' is for polyfill
    if (ciuresult == 'n' || ciuresult == 'p') return false;
    throw 'unknown return value from can i use';
  }

  function testify(o){

    var ciubool = bool(o.ciuresult);

    // caniuse says audio/video are yes/no, Modernizr has more detail which we'll dumb down.
    if (~TEST.audvid.indexOf(o.feature))
      o.result = !!o.result;

    // if caniuse gave us a 'partial', lets let it pass with a note.
    if (o.ciuresult == 'a'){
      return ok(true,
        o.browser + o.version + ': Caniuse reported partial support for ' + o.ciufeature +
        '. So.. Modernizr\'s ' + o.result + ' is good enough...'
      );
    }


    // change the *documented* false positives
    if ((o.feature == 'textshadow' && o.browser == 'firefox' && o.version == 3)
        && ciubool == false
    ) {
      ciubool = o.fp = true;
    }

    // where we actually do most our assertions
    equal(o.result, ciubool,
      o.browser + o.version + ': Caniuse result for ' + o.ciufeature +
      ' matches Modernizr\'s ' + (o.fp ? '*false positive*' : 'result') + ' for ' + o.feature
    );
  }


  module('caniuse.com data matches', {
      setup:function() {
      },
      teardown:function() {
      }
  });


  test("we match caniuse data", function() {

    for (var feature in Modernizr){

      var ciufeatname = map[feature];

      if (ciufeatname === undefined) continue;

      var ciufeatdata = testdata[ciufeatname];

      if (ciufeatdata === undefined) throw 'unknown key of caniusedata';

      // get results for this feature for all versions of this browser
      var browserResults = ciufeatdata.stats[currBrowser];

      // let's get our versions in order..
      var minorver   = ua.minor &&                                  // caniuse doesn't use two digit minors
                       ua.minor.toString().replace(/(\d)\d$/,'$1'), // but opera does.

          majorminor = (ua.major + '.' + minorver)
                          // opera gets grouped in some cases by caniuse
                          .replace(/(9\.(6|5))/ , ua.family == 'opera' ? '9.5-9.6'   : "$1")
                          .replace(/(10\.(0|1))/, ua.family == 'opera' ? '10.0-10.1' : "$1"),

          mmResult   = browserResults[majorminor],
          mResult    = browserResults[ua.major];


      // check it against the major.minor: eg. FF 3.6
      if (mmResult && mmResult != 'u'){ // 'y' 'n' or 'a'

        // data ends w/ ` x` if its still prefixed in the imp
        mmResult = mmResult.replace(' x','');

        // match it against our data.
        testify({ feature     : feature
                , ciufeature  : ciufeatname
                , result      : Modernizr[feature]
                , ciuresult   : mmResult
                , browser     : currBrowser
                , version     : majorminor
        });

        continue; // don't check the major version
      }

      // check it against just the major version: eg. FF 3
      if (mResult){

        // unknown support from caniuse... He would probably like to know our data, though!
        if (mResult == 'u') continue;

        // data ends w/ ` x` if its still prefixed in the imp
        mResult = mResult.replace(' x','');

        testify({ feature     : feature
                , ciufeature  : ciufeatname
                , result      : Modernizr[feature]
                , ciuresult   : mResult
                , browser     : currBrowser
                , version     : ua.major
        });


      }

    } // for in loop

  }); // eo test()


}; // eo caniusecallback()
