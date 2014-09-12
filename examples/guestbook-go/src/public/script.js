$( document ).ready(function() {

  appendGuestbookEntries = function( data ) {
    $( "#guestbook-entries" ).empty();
    $.each( data, function( key, val ) {
      $( "#guestbook-entries" ).append( "<p>" + val + "</p>" );
    });
  }

  handleSubmission = function() {
    value = $( "#guestbook-entry-content" ).val()
    if (value.length > 0) {
      $( "#guestbook-entries" ).append( "<p>...</p>" );
      $.getJSON( "rpush/guestbook/" + value, appendGuestbookEntries);
    }
    return false;
  }

  // Event handlers.
  $( "#guestbook-submit" ).click(handleSubmission);
  $( "#guestbook-entry-content" ).keypress(function (e) {
    if (e.which == 13) {
      return handleSubmission();
    }
  });

  // Poll every second.
  (function fetchGuestbook(){

    $.getJSON("lrange/guestbook").done(appendGuestbookEntries).always(function() { setTimeout(fetchGuestbook, 1000); });

  })();

});
