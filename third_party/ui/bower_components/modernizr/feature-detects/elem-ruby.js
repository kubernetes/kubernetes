// Browser support test for the HTML5 <ruby>, <rt> and <rp> elements
// http://www.whatwg.org/specs/web-apps/current-work/multipage/text-level-semantics.html#the-ruby-element
//
// by @alrra

Modernizr.addTest('ruby', function () {

    var ruby = document.createElement('ruby'),
        rt = document.createElement('rt'),
        rp = document.createElement('rp'),
        docElement = document.documentElement,
        displayStyleProperty = 'display',
        fontSizeStyleProperty = 'fontSize'; // 'fontSize' - because it`s only used for IE6 and IE7

    ruby.appendChild(rp);
    ruby.appendChild(rt);
    docElement.appendChild(ruby);

    // browsers that support <ruby> hide the <rp> via "display:none"
    if ( getStyle(rp, displayStyleProperty) == 'none' ||                                                       // for non-IE browsers
    // but in IE browsers <rp> has "display:inline" so, the test needs other conditions:
        getStyle(ruby, displayStyleProperty) == 'ruby' && getStyle(rt, displayStyleProperty) == 'ruby-text' || // for IE8 & IE9
        getStyle(rp, fontSizeStyleProperty) == '6pt' && getStyle(rt, fontSizeStyleProperty) == '6pt' ) {       // for IE6 & IE7

        cleanUp();
        return true;

    } else {
        cleanUp();
        return false;
    }

    function getStyle( element, styleProperty ) {
        var result;

        if ( window.getComputedStyle ) {     // for non-IE browsers
            result = document.defaultView.getComputedStyle(element,null).getPropertyValue(styleProperty);
        } else if ( element.currentStyle ) { // for IE
            result = element.currentStyle[styleProperty];
        }

        return result;
    }

    function cleanUp() {
        docElement.removeChild(ruby);
        // the removed child node still exists in memory, so ...
        ruby = null;
        rt = null;
        rp = null;
    }

});
