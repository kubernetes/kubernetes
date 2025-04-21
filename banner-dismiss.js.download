$(document).ready(function() {
  function setCookie(name, value, days) {
    let expires = "";
    let date = new Date(); // Create a new Date object
    let dateToSecond = 24 * 60 * 60 * 1000;

    if (days) {
      date.setTime(date.getTime() + days * dateToSecond); // Modify the existing Date object
      expires = "; expires=" + date.toUTCString();
    }

    document.cookie = name + "=" + value + expires + "; path=/";
  }

  function getCookie(name) {
    let matches = document.cookie.match(new RegExp(
      "(?:^|; )" + name.replace(/([\.$?*|{}\(\)\[\]\\\/\+^])/g, '\\$1') + "=([^;]*)"
    ));
    return matches ? "true" : undefined;
  }

  /* Check the presence of a cookie */
  let announcement = document.querySelector("#announcement");
  let token = `announcement_ack_${announcement.getAttribute('data-announcement-name').replace(/\s/g, '_')}`; // Generate the unique token for announcement
  let acknowledged = getCookie(token);
  if (acknowledged === "true") {
    announcement.remove(); // Remove the announcement if the cookie is set
  }
  else {
    announcement.classList.add('display-announcement') // Display the announcement if the cookie is not set
  }

  /* Driver code to set the cookie */
  let button = document.querySelector('#banner-dismiss');
  button.removeAttribute('style');
  button.addEventListener('click', function() {
    setCookie(token, "true", 
    button.getAttribute('data-ttl')); // Set a cookie with time to live parameter
    announcement.remove();
  });
});
