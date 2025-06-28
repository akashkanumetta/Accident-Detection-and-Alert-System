document.addEventListener("DOMContentLoaded", function () {
    function fetchAccidentData() {
        fetch("http://127.0.0.1:5500/get-accident-data")
            .then(response => response.json())
            .then(data => {
                const alertBox = document.getElementById("alert-box");
                const tableBody = document.getElementById("accident-table");

                tableBody.innerHTML = "";

                if (data.accidents.length > 0) {
                    alertBox.style.display = "block";
                    alertBox.innerHTML = `<p>⚠️ Accident Detected!</p>`;
                } else {
                    alertBox.style.display = "none";
                }

                data.accidents.forEach(accident => {
                    let row = `<tr>
                        <td>${accident.time}</td>
                        <td>${accident.date}</td>
                        <td>${accident.location}</td>
                        <td><img src="${accident.snapshot}" width="100"></td>
                    </tr>`;
                    tableBody.innerHTML += row;
                });
            });
    }

    setInterval(fetchAccidentData, 5000);
});
