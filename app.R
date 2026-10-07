library(shiny)

library(recipes)

library(klaR)

source("predict.R", local = TRUE)

bundle <- readRDS("model_bundle.rds")

metadata <- bundle$feature_metadata

stopifnot(identical(metadata$model_variable, bundle$final_features))

controls <- lapply(seq_len(nrow(metadata)), function(i) {
    item <- metadata[i, ]
    label <- paste0(item$english_name, if (nzchar(item$unit)) 
        paste0(" (", item$unit, ")")
    else "")
    if (item$type == "continuous") {
        return(numericInput(paste0("v", i), label, value = NA_real_, min = item$allowed_min, max = item$allowed_max))
    }
    selectInput(paste0("v", i), label, choices = c("", levels(bundle$input_schema[[item$model_variable]])))
})

ui <- fluidPage(tags$head(tags$style(HTML("body{background:#f3f6f8;color:#243447;font-family:Arial,sans-serif}\n     .container-fluid{max-width:1000px;padding:24px}\n     h2{color:#123b5d}.well{background:white;border-radius:10px}\n     .btn-primary{background:#176b87;border-color:#176b87}\n     #probability{color:#c34f32;font-size:40px;font-weight:bold}"))), 
    titlePanel("Prediction Prototype"), sidebarLayout(sidebarPanel(controls, actionButton("calculate", 
        "Calculate probability", class = "btn-primary")), mainPanel(h4("Predicted outcome probability"), 
        textOutput("probability"), p("Research prototype; not for standalone clinical decisions."))))

server <- function(input, output, session) {
    probability <- eventReactive(input$calculate, {
        row <- bundle$input_schema[NA_integer_, , drop = FALSE]
        for (i in seq_len(nrow(metadata))) {
            item <- metadata[i, ]
            variable <- item$model_variable
            value <- input[[paste0("v", i)]]
            if (item$type == "continuous") {
                value <- as.numeric(value)
                validate(need(length(value) == 1L && is.finite(value), paste(item$english_name, "is required.")))
                validate(need(value >= item$allowed_min && value <= item$allowed_max, paste(item$english_name, 
                  "is outside the allowed range.")))
            }
            else {
                validate(need(nzchar(value), paste(item$english_name, "is required.")))
                value <- factor(value, levels = levels(bundle$input_schema[[variable]]))
                validate(need(!is.na(value), "Invalid categorical value."))
            }
            row[[variable]] <- value
        }
        p <- predict_bundle(bundle, row)
        validate(need(length(p) == 1L && is.finite(p) && p >= 0 && p <= 1, "Invalid prediction."))
        p
    }, ignoreInit = TRUE)
    output$probability <- renderText(sprintf("%.1f%%", 100 * probability()))
}

shinyApp(ui, server)
