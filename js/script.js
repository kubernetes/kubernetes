$(document).ready(function (){

    init_nav_bar(jQuery);
    init_back_to_top(jQuery);
    init_youtube_fix_ie(jQuery);
    init_mobile_menu_button(jQuery);

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
        },500);
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
