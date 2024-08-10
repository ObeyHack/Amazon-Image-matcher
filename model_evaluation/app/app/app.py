"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx

from rxconfig import config



def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.center(
        (rx.vstack(
        rx.upload(
            rx.text(
                "Drag and drop files here or click to select files"
            ),
            id="my_upload",
            border="1px dotted rgb(107,99,246)",
            padding="5em",
                )
            )
        )
    )



app = rx.App()
app.add_page(index)
