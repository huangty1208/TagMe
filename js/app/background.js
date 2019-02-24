console.log("background script loaded.");

// Listen for received message from content script and set to background scope globally
chrome.runtime.onMessage.addListener(receiver);

var selectedText = "";

function receiver(message, sender, sendResponse) {
  console.log("message Received: " + message);
  selectedText = message;


   $.ajax({
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ "content" : selectedText}),
                dataType: 'json',
				async: false,
                url: "http://www.tagme.icu:5000/_process",
                success: function (result) {
                    console.log("success"+ result["tag"]);
                    selectedText = result["tag"]
                },
                error: function() {
                console.log("error");
            }
    });
  
  
}

