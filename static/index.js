
    document.getElementById("submitBtn").addEventListener("click", function() {
        const inputValue = document.getElementById("fname").value;
       //alert(inputValue); 
        const formData = new FormData();
        formData.append("file", document.getElementById("file").files[0]);
        formData.append("text", document.getElementById("fname").value);
    
        fetch("/ask", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Answer:", data.answer);
            alert("Answer:" + data.answer);
        });
        
    
    });

   