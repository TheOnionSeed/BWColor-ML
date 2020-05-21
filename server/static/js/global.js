$( document ).ready(function() {

    // UPLOAD CLASS DEFINITION
    // ======================

    var dropZone = document.getElementById('drop-zone');

    var startUpload = function(files) {
        if(files.length >1){
           alert("Please upload 1 file only!");
        }else{
           var fd= new FormData();
           fd.append('imagefile',files[0]);
           
           $.ajax({
            url:'uploader',
            type:'post',
            data:fd,
            contentType:false,
            processData:false,
            success:function(response){
               if(response!==0){
                  //console.log("success" + response["Image"]);
                  $('#imgConv').html('<img src="data:image/png;base64,' + response["Image"]  + '" style="height: 50%; width:50%; margin-top:2em;" />');
               }
               else{
                  console.log("fail" + response);
               }
            }
            
           });
        }
    }

    dropZone.ondrop = function(e) {
        e.preventDefault();
        this.className = 'upload-drop-zone';

        startUpload(e.dataTransfer.files)
    }

    dropZone.ondragover = function() {
        this.className = 'upload-drop-zone drop';
        return false;
    }

    dropZone.ondragleave = function() {
        this.className = 'upload-drop-zone';
        return false;
    }

});