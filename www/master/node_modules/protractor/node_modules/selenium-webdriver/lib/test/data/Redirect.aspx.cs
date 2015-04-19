using System;

public partial class Redirect : Page
{
    protected new void Page_Load(object sender, EventArgs e)
    {
        Response.Redirect("resultPage.html");
    }
}
