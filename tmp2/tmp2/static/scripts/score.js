$(function () {
    console.log("score.js ready!");
});

$("#predict").click(function (event) {
    event.preventDefault();
    // indicate calculation in progress
    $(".result").html("...");
    var comment = $("textarea#comment").val();
    var delivery_ok = $("input#delivery").prop('checked') === true ? 'yes' : 'no';
    var deliveryday = $("input#deliveryday").val();
    var region = $("input#region").val();

    var pos = $("input#pos").prop('checked');
    var lemma = $("input#lemma").prop('checked');
    var bow = $("input#bag").prop('checked');
    var vader = $("input#vader").prop('checked');
    var ml_tech = $('input[name=mlGroup]:checked').val();

    if (!bow && !vader) {
        window.alert("Either bag-of-words or Sentiment Analysis has to be selected")
        return
    }

    if (!deliveryday) {
        deliveryday = 0;
    }
    if (!region) {
        region = 0;
    }

    if (comment === "") {
        $("span#result").html("");
    } else {
        $.ajax({
            url: "/score/result",
            type: "POST",
            data: {
                comment: comment,
                delivery_ok: delivery_ok,
                deliveryday: deliveryday,
                region: region,
                pos: pos,
                lemma: lemma,
                bow: bow,
                vader: vader,
                ml_tech: ml_tech
            },
            success: function (data) {
                $("span#result").html(data.result);
            },
            error: function () {
                $("span#result").html("prediction failed, please retry");
            }
        });
    }
});

$("#save_option").click(function (event) {
    event.preventDefault();
    // indicate calculation in progress

    var nlp_pos = $("input#pos").prop('checked');
    var nlp_lemma = $("input#lemma").prop('checked');
    var nlp_bag = $("input#bag").prop('checked');
    var nlp_vader = $("input#vader").prop('checked');
    var ml_option = $("input:radio[name='mlGroup']:checked").val();

    $('#predict').prop('disabled', true);
    $('#submit').prop('disabled', true);
    $("#result").html("");
    var overlay = jQuery('<div id="overlay"><div class="progress">\n' +
        '  <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width: 100%"></div>\n' +
        '</div></div>');
    overlay.appendTo(document.body);
    $.ajax({
        url: "/score/option",
        type: "POST",
        data: {
            pos: nlp_pos,
            lemma: nlp_lemma,
            bag: nlp_bag,
            vader: nlp_vader,
            ml_option: ml_option
        },
        success: function (data) {
            $('#predict').prop('disabled', false);
            $('#submit').prop('disabled', false);
            $("#overlay").remove();
        },
        error: function () {
            alert("Training fails, please retrain the instances");
            $("#overlay").remove();
        }
    });
});

$(function () {
    $('input[type="file"]').change(function (e) {
        var fileName = e.target.files[0].name;
        $("#file_upload").html(fileName);
    });
});