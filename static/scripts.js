document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll("form").forEach(form => {
        form.addEventListener('submit', function() {
            document.getElementById("loadingOverlay").style.display = "block";
        });
    });

    document.querySelectorAll("a.show-plot").forEach(link => {
        link.addEventListener('click', function() {
            document.getElementById("loadingOverlay").style.display = "block";
        });
    });

    document.querySelectorAll('img.enlargeable').forEach(img => {
        img.addEventListener('click', () => {
            document.getElementById('modalImage').src = img.src;
            document.getElementById('imgModal').style.display = "block";
        });
    });

    document.querySelector('.close').addEventListener('click', () => {
        document.getElementById('imgModal').style.display = "none";
    });

    window.onclick = function(event) {
        const modal = document.getElementById('imgModal');
        if (event.target === modal) {
            modal.style.display = "none";
        }
    };
});
