//modal close button
(function(){
    //π.modalCloseButton = function(closingFunction){
    //	return π.button('pi-modal-close-button', null, null, closingFunction);
    //};
})();

// globals
var body;

//helper functions
function booleanAttributeValue(element, attribute, defaultValue){
    // returns true if an attribute is present with no value
    // e.g. booleanAttributeValue(element, 'data-modal', false);
    if (element.hasAttribute(attribute)) {
        var value = element.getAttribute(attribute);
        if (value === '' || value === 'true') {
            return true;
        } else if (value === 'false') {
            return false;
        }
    }

    return defaultValue;
}

function classOnCondition(element, className, condition) {
    if (condition)
        $(element).addClass(className);
    else
        $(element).removeClass(className);
}

function highestZ() {
    var Z = 1000;

    $("*").each(function(){
        var thisZ = $(this).css('z-index');

        if (thisZ != "auto" && thisZ > Z) Z = ++thisZ;
    });

    return Z;
}

function newDOMElement(tag, className, id){
    var el = document.createElement(tag);

    if (className) el.className = className;
    if (id) el.id = id;

    return el;
}

function px(n){
    return n + 'px';
}

var kub = (function () {
    var HEADER_HEIGHT;
    var html, header, mainNav, quickstartButton, hero, encyclopedia, footer, headlineWrapper;

    $(document).ready(function () {
        html = $('html');
        body = $('body');
        header = $('header');
        mainNav = $('#mainNav');
        quickstartButton = $('#quickstartButton');
        hero = $('#hero');
        encyclopedia = $('#encyclopedia');
        footer = $('footer');
        headlineWrapper = $('#headlineWrapper');
        HEADER_HEIGHT = header.outerHeight();

        document.documentElement.classList.remove('no-js');

        resetTheView();

        window.addEventListener('resize', resetTheView);
        window.addEventListener('scroll', resetTheView);
        window.addEventListener('keydown', handleKeystrokes);

        document.onunload = function(){
            window.removeEventListener('resize', resetTheView);
            window.removeEventListener('scroll', resetTheView);
            window.removeEventListener('keydown', handleKeystrokes);
        };

        setInterval(setFooterType, 10);
    });

    function setFooterType() {
        var windowHeight = window.innerHeight;
        var bodyHeight;

        switch (html[0].id) {
            case 'docs': {
                bodyHeight = hero.outerHeight() + encyclopedia.outerHeight();
                break;
            }

            case 'home':
            // case 'caseStudies':
                bodyHeight = windowHeight;
                break;
            case 'blog':
                bodyHeight = windowHeight;
            case 'caseStudies':
            case 'partners':
                bodyHeight = windowHeight * 2;
                break;

            default: {
                bodyHeight = hero.outerHeight() + $('#mainContent').outerHeight();
            }
        }

        var footerHeight = footer.outerHeight();
        classOnCondition(body, 'fixed', windowHeight - footerHeight > bodyHeight);
    }

    function resetTheView() {
        if (html.hasClass('open-nav')) {
            toggleMenu();
        } else {
            HEADER_HEIGHT = header.outerHeight();
        }

        if (html.hasClass('open-toc')) {
            toggleToc();
        }

        classOnCondition(html, 'flip-nav', window.pageYOffset > 0);

        if (html[0].id == 'home') {
            setHomeHeaderStyles();
        }
    }

    function setHomeHeaderStyles() {
        if (!quickstartButton[0]) {
            return;
        }
        var Y = window.pageYOffset;
        var quickstartBottom = quickstartButton[0].getBoundingClientRect().bottom;

        classOnCondition(html[0], 'y-enough', Y > quickstartBottom);
    }

    function toggleMenu() {
        if (window.innerWidth < 800) {
            pushmenu.show('primary');
        }

        else {
            var newHeight = HEADER_HEIGHT;

            if (!html.hasClass('open-nav')) {
                newHeight = mainNav.outerHeight();
            }

            header.css({height: px(newHeight)});
            html.toggleClass('open-nav');
        }
    }

    function handleKeystrokes(e) {
        switch (e.which) {
            case 27: {
                if (html.hasClass('open-nav')) {
                    toggleMenu();
                }
                break;
            }
        }
    }

    function showVideo() {
        $('body').css({overflow: 'hidden'});

        var videoPlayer = $("#videoPlayer");
        var videoIframe = videoPlayer.find("iframe")[0];
        videoIframe.src = videoIframe.getAttribute("data-url");
        videoPlayer.css({zIndex: highestZ()});
        videoPlayer.fadeIn(300);
        videoPlayer.click(function(){
            $('body').css({overflow: 'auto'});

            videoPlayer.fadeOut(300, function(){
                videoIframe.src = '';
            });
        });
    }

    function tocWasClicked(e) {
        var target = $(e.target);
        var docsToc = $("#docsToc");
        return (target[0] === docsToc[0] || target.parents("#docsToc").length > 0);
    }

    function listenForTocClick(e) {
        if (!tocWasClicked(e)) toggleToc();
    }

    function toggleToc() {
        html.toggleClass('open-toc');

        setTimeout(function () {
            if (html.hasClass('open-toc')) {
                window.addEventListener('click', listenForTocClick);
            } else {
                window.removeEventListener('click', listenForTocClick);
            }
        }, 100);
    }

    return {
        toggleToc: toggleToc,
        toggleMenu: toggleMenu,
        showVideo: showVideo
    };
})();


// accordion
(function(){
    var yah = true;
    var moving = false;
    var CSS_BROWSER_HACK_DELAY = 25;

    $(document).ready(function(){
        // Safari chokes on the animation here, so...
        if (navigator.userAgent.indexOf('Chrome') == -1 && navigator.userAgent.indexOf('Safari') != -1){
            var hackStyle = newDOMElement('style');
            hackStyle.innerHTML = '.pi-accordion .wrapper{transition: none}';
            body.append(hackStyle);
        }
        // Gross.

        $('.pi-accordion').each(function () {
            var accordion = this;
            var content = this.innerHTML;
            var container = newDOMElement('div', 'container');
            container.innerHTML = content;
            $(accordion).empty();
            accordion.appendChild(container);
            CollapseBox($(container));
        });

        setYAH();

        setTimeout(function () {
            yah = false;
        }, 500);
    });

    function CollapseBox(container){
        container.children('.item').each(function(){
            // build the TOC DOM
            // the animated open/close is enabled by having each item's content exist in the flow, at its natural height,
            // enclosed in a wrapper with height = 0 when closed, and height = contentHeight when open.
            var item = this;

            // only add content wrappers to containers, not to links
            var isContainer = item.tagName === 'DIV';

            var titleText = item.getAttribute('data-title');
            var title = newDOMElement('div', 'title');
            title.innerHTML = titleText;

            var wrapper, content;

            if (isContainer) {
                wrapper = newDOMElement('div', 'wrapper');
                content = newDOMElement('div', 'content');
                content.innerHTML = item.innerHTML;
                wrapper.appendChild(content);
            }

            item.innerHTML = '';
            item.appendChild(title);

            if (wrapper) {
                item.appendChild(wrapper);
                $(wrapper).css({height: 0});
            }


            $(title).click(function(){
                if (!yah) {
                    if (moving) return;
                    moving = true;
                }

                if (container[0].getAttribute('data-single')) {
                    var openSiblings = item.siblings().filter(function(sib){return sib.hasClass('on');});
                    openSiblings.forEach(function(sibling){
                        toggleItem(sibling);
                    });
                }

                setTimeout(function(){
                    if (!isContainer) {
                        moving = false;
                        return;
                    }
                    toggleItem(item);
                }, CSS_BROWSER_HACK_DELAY);
            });

            function toggleItem(thisItem){
                var thisWrapper = $(thisItem).find('.wrapper').eq(0);

                if (!thisWrapper) return;

                var contentHeight = thisWrapper.find('.content').eq(0).innerHeight() + 'px';

                if ($(thisItem).hasClass('on')) {
                    thisWrapper.css({height: contentHeight});
                    $(thisItem).removeClass('on');

                    setTimeout(function(){
                        thisWrapper.css({height: 0});
                        moving = false;
                    }, CSS_BROWSER_HACK_DELAY);
                } else {
                    $(item).addClass('on');
                    thisWrapper.css({height: contentHeight});

                    var duration = parseFloat(getComputedStyle(thisWrapper[0]).transitionDuration) * 1000;

                    setTimeout(function(){
                        thisWrapper.css({height: ''});
                        moving = false;
                    }, duration);
                }
            }

            if (content) {
                var innerContainers = $(content).children('.container');
                if (innerContainers.length > 0) {
                    innerContainers.each(function(){
                        CollapseBox($(this));
                    });
                }
            }
        });
    }

    function setYAH() {
        var pathname = location.href.split('#')[0]; // on page load, make sure the page is YAH even if there's a hash
        var currentLinks = [];

        $('.pi-accordion a').each(function () {
            if (pathname === this.href) currentLinks.push(this);
        });

        currentLinks.forEach(function (yahLink) {
            $(yahLink).parents('.item').each(function(){
                $(this).addClass('on');
                $(this).find('.wrapper').eq(0).css({height: 'auto'});
                $(this).find('.content').eq(0).css({opacity: 1});
            });

            $(yahLink).addClass('yah');
            yahLink.onclick = function(e){e.preventDefault();};
        });
    }
})();


var pushmenu = (function(){
    var allPushMenus = {};

    $(document).ready(function(){
        $('[data-auto-burger]').each(function(){
            var container = this;
            var id = container.getAttribute('data-auto-burger');

            var autoBurger = document.getElementById(id) || newDOMElement('div', 'pi-pushmenu', id);
            var ul = autoBurger.querySelector('ul') || newDOMElement('ul');

            $(container).find('a[href], button').each(function () {
                if (!booleanAttributeValue(this, 'data-auto-burger-exclude', false)) {
                    var clone = this.cloneNode(true);
                    clone.id = '';

                    if (clone.tagName == "BUTTON") {
                        var aTag = newDOMElement('a');
                        aTag.href = '';
                        aTag.innerHTML = clone.innerHTML;
                        aTag.onclick = clone.onclick;
                        clone = aTag;
                    }
                    var li = newDOMElement('li');
                    li.appendChild(clone);
                    ul.appendChild(li);
                }
            });

            autoBurger.appendChild(ul);
            body.append(autoBurger);
        });

        $(".pi-pushmenu").each(function(){
            allPushMenus[this.id] = PushMenu(this);
        });
    });

    function show(objId) {
        allPushMenus[objId].expose();
    }

    function PushMenu(el) {
        var html = document.querySelector('html');

        var overlay = newDOMElement('div', 'overlay');
        var content = newDOMElement('div', 'content');
        content.appendChild(el.querySelector('*'));

        var side = el.getAttribute("data-side") || "right";

        var sled = newDOMElement('div', 'sled');
        $(sled).css(side, 0);

        sled.appendChild(content);

        var closeButton = newDOMElement('button', 'push-menu-close-button');
        closeButton.onclick = closeMe;

        sled.appendChild(closeButton);

        overlay.appendChild(sled);
        el.innerHTML = '';
        el.appendChild(overlay);

        sled.onclick = function(e){
            e.stopPropagation();
        };

        overlay.onclick = closeMe;

        window.addEventListener('resize', closeMe);

        function closeMe(e) {
            if (e.target == sled) return;

            $(el).removeClass('on');
            setTimeout(function(){
                $(el).css({display: 'none'});

                $(body).removeClass('overlay-on');
            }, 300);
        }

        function exposeMe(){
            $(body).addClass('overlay-on'); // in the default config, kills body scrolling

            $(el).css({
                display: 'block',
                zIndex: highestZ()
            });

            setTimeout(function(){
                $(el).addClass('on');
            }, 10);
        }

        return {
            expose: exposeMe
        };
    }

    return {
        show: show
    };
})();

$(function() {
    // If vendor strip doesn't exist add className
    if ( !$('#vendorStrip').length > 0 ) {
        $('.header-hero').addClass('bot-bar');
    }

    // If is not homepage add class to hero section
    if (!$('.td-home').length > 0 ) {
        $('.header-hero').addClass('no-sub');
    }
});
