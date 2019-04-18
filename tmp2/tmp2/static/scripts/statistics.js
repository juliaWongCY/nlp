$(function () {
    console.log("statistics.js ready!");
});

var error_msg = "error, please recalculate";


$("#refresh").click(function (event) {
    event.preventDefault();
    // indicate calculation in progress
    $(".stat").html("recalculating");
    var features = get_features();
    if (features === '') {
        alert("Error, must choose as least one feature for analysis")
        console.log("error in posting refresh request to pyramid");
        $(".stat").html(error_msg);
    } else {
        $.ajax(
            {
                url: "statistics/refresh",
                type: "POST",
                data: {
                    action: "refresh",
                    features: features
                },
                success: function (data) {
                    $("#dtree_a").html(data.dtree_a);
                    $("#svm_a").html(data.svm_a);
                    $("#log_reg_a").html(data.log_reg_a);
                    $("#k_a").html(data.k_a);
                    $("#nb_a").html(data.nb_a);
                    $("#rf_a").html(data.rf_a);
                    $("#dtree_f").html(data.dtree_f);
                    $("#svm_f").html(data.svm_f);
                    $("#log_reg_f").html(data.log_reg_f);
                    $("#k_f").html(data.k_f);
                    $("#nb_f").html(data.nb_f);
                    $("#rf_f").html(data.rf_f);
                },
                error: function () {
                    $(".stat").html(error_msg);
                }
            });
    }
});

function get_features() {
    var features = '';
    if ($('#c_neu').is(":checked")) {
        features += 'neu '
    }
    if ($('#c_neg').is(":checked")) {
        features += 'neg '
    }
    if ($('#c_pos').is(":checked")) {
        features += 'pos '
    }
    if ($('#c_compound').is(":checked")) {
        features += 'compound '
    }
    if ($('#c_ontime').is(":checked")) {
        features += 'on_time_in_full '
    }
    if ($('#c_day').is(":checked")) {
        features += 'deliveryday '
    }
    if ($('#c_region').is(":checked")) {
        features += 'region '
    }
    if ($('#c_bow').is(":checked")) {
        features += 'bag_vector '
    }

    return features;
};
