$(document).ready(function() {
    initDropzone();
    $('#classify-btn').click(submitImage);
});

function initDropzone() {
    Dropzone.options.imageDropzone = {
        url: '/predict',
        autoProcessQueue: false,
        addRemoveLinks: true,
        init: function() {
            this.on("addedfile", function(file) {
                console.log("File added: ", file.name);
            });
            this.on("removedfile", function(file) {
                console.log("File removed: ", file.name);
            });
        }
    };
}

function submitImage() {
    const dropzone = Dropzone.forElement("#image-dropzone");
    if (dropzone.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }
    
    const formData = new FormData();
    formData.append('image', dropzone.files[0]);

    $.ajax({
        url: '/predict',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            $('#result').text(`Class ID: ${response.class_id}, Class Name: ${response.class_name}`);
            $('#result-holder').show();
        },
        error: function(xhr, status, error) {
            console.error("Error: ", error);
            $('#result').text('Error in classification.');
            $('#result-holder').show();
        }
    });
}
