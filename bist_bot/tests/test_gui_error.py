from bist_bot.gui_app import BistBotWindow


def test_gui_error_handler_sets_status_message(qtbot):
    window = BistBotWindow()
    qtbot.addWidget(window)

    window._on_error("Test error")
    assert "Test error" in window.statusBar().currentMessage()
