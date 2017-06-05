$(document).ready(function() {
    var codeTabs = $('#code-tabs-wrapper').find('li');

    for (var i = 0; i < codeTabs.length; i++) {
        createCodeTabListeners(codeTabs, i);
    }

    function createCodeTabListeners(codeTabs, index) {
        var tab = $(codeTabs[index]),
            id = tab.attr('id'),
            codeClass = '.' + id;
        tab.on('click', function() {
            codeTabs.removeClass('tab-selected');
            tab.addClass('tab-selected');
            $('.code-block').removeClass('active');
            $(codeClass).addClass('active');
         
        });
    }

    function setDefautTab() {
        $(codeTabs[0]).addClass('tab-selected');
        $('.' + codeTabs[0].id).addClass('active');
    }

    setDefautTab();
});