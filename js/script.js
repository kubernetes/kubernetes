$(document).ready(function (){

    init_nav_bar(jQuery);
    init_back_to_top(jQuery);
    init_youtube_fix_ie(jQuery);
    init_mobile_menu_button(jQuery);
    init_browser_detect(jQuery);

});

function init_nav_bar($) {
    var w = $(window),
        n = $('#nav'),
        c = 'alt';

    var has_class = false;

    w.scroll(function() {
        var s = w.scrollTop();
        if ( 0 < s ) {
            if ( !has_class )
                on();
        } else {
            if ( has_class ) 
                off();
        }
    });

    function on() {
        n.addClass(c);
        has_class = true;
    }

    function off() {
        n.removeClass(c);
        has_class = false;
    }
}

function init_back_to_top($) {
    $('.back-to-top').click(function(e) {
        e.preventDefault();
        $('html,body').animate({
            scrollTop: 0 
        },{
            duration: 500,
            done: function() {
                // Hack for Windows Phone
                setTimeout(function() { 
                    window.scrollTo(0,0);
                },550);
            }
        });
    });
}

function init_youtube_fix_ie($) {
    $('iframe').each(function(){
        var url = $(this).attr("src");
        $(this).attr("src",url+"&wmode=transparent");
    });
}

function init_mobile_menu_button($) {
    $('#mobile-menu-button').on('click',function(e) {
        e.preventDefault();
        $('.mobile-menu-slide').toggleClass('slide-in');
    });
}

/*
 * Seemed like a bit of overkill to make a jQuery plugin for this, so 
 * I'm just passing in the selector.
 */
function init_list_load_more($,selector,num) {
    var i = 0,
        l = num || 1,
        to_show = num || 1,
        items = $('ul.list li');

    function show() {
        for ( i; i < l; i++ ) {
            items.eq(i).fadeIn('fast');
        }
        l += to_show;
        if ( i >= items.length ) {
            selector.fadeOut();
        }
    }

    selector.on('click',function() {
        show();
    });

    return show();
}

function is_responsive_size(size) {
    var selector = '';
    switch ( size ) {
        case 'xs':
            selector = '.visible-xs-block';
            break;
        case 'sm':
            selector = '.visible-sm-block';
            break;
        case 'md':
            selector = '.visible-md-block';
            break;
        case 'lg':
            selector = '.visible-lg-block';
            break;
    }
    if ( 'none' != $(selector).css('display') ) {
        return true;
    } else {
        return false;
    }

}

var is_xs = function() {
    return is_responsive_size('xs');
}

var is_sm = function() {
    return is_responsive_size('sm');
}

var is_md = function() {
    return is_responsive_size('md');
}

var is_lg = function() {
    return is_responsive_size('lg');
}

function init_browser_detect($) {
    /* 
     * Currently only used to detect IE Mobile. Might use for IE10...
     */
    function detect() {
        if ( navigator.userAgent.match(/Windows Phone/i) ) {
            $('body').addClass('ie-mobile');
        }
    }

    return detect();
}

