// https://jsfiddle.net/upqwhou2/

$(document).ready(function() {
    var navigationLinks = $('#sidebar-wrapper > ul li a');
    var navigationSections = $('#sidebar-wrapper > ul > ul');
    var sectionIdTonavigationLink = {};
    var sections = $('#page-content-wrapper').find('h1, h2').map(function(index, node) {
        if (node.id) {
            sectionIdTonavigationLink[node.id] = $('#sidebar-wrapper > ul li a[href="#' + node.id + '"]');
            return node;
        }
    });
    var sectionIdToNavContainerLink = {};
    var topLevelSections = $('#page-content-wrapper').find('h1').map(function(index, node) {
        if (node.id) {
            sectionIdToNavContainerLink[node.id] = $('#sidebar-wrapper > ul > ul[id="' + node.id + '-nav' +'"]');
            return node;
        }
    });

    var firstLevelNavs = $('#sidebar-wrapper > li');
    var secondLevelNavs = $('#sidebar-wrapper > ul > ul');
    var secondLevelNavContents = $('#sidebar-wrapper > ul > ul > li');
    var thirdLevelNavs = null; // TODO: When compile provides 3 level nav, implement

    var sectionsReversed = $(sections.get().reverse());

    function checkScroll(event) {
        var scrollPosition = $(window).scrollTop();
        var offset = 50;
        scrollPosition += offset;
        sections.each(function() {
            var currentSection = $(this);
            var sectionTop = $(this).offset().top;
            var id = $(this).attr('id');
            if (scrollPosition >= sectionTop) {
                navigationLinks.removeClass('selected');
                sectionIdTonavigationLink[id].addClass('selected');
                var sectionNavContainer = sectionIdToNavContainerLink[id];
                var sectionNavContainerDisplay;
                if (sectionNavContainer) {
                    sectionNavContainerDisplay = sectionNavContainer.css('display');
                }
                if (sectionNavContainer && sectionNavContainerDisplay === 'none') {
                    navigationSections.toggle(false);
                    sectionNavContainer.toggle(true);
                }
            }
            if (($(this).offset().top < window.pageYOffset + 50) && $(this).offset().top + $(this).height() > window.pageYOffset) {
                window.location.hash = id;
            }
        });
    }
    checkScroll();
    $(window).on('scroll', function(event) {
        checkScroll(event);
    });
});