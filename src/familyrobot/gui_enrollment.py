"""PySide6 GUI for local identity enrollment."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from familyrobot.enrollment import (
    ENROLLMENT_IMAGES_DIR,
    EnrollmentManifest,
    EnrollmentRecord,
    EnrollmentStore,
    default_enrollment_root,
)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _sanitize_file_name(file_name: str) -> str:
    return file_name.replace("/", "_").replace("\\", "_")


def _unique_destination(directory: Path, file_name: str) -> Path:
    destination = directory / _sanitize_file_name(file_name)
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _format_family_examples() -> str:
    return (
        "示例：身份编号可写 father / mother / child_1 / grandma；"
        "显示名称可写 爸爸 / 妈妈 / 小明 / 奶奶；"
        "角色可写 父亲 / 母亲 / 儿子 / 祖母。"
    )


@dataclass(frozen=True, slots=True)
class EnrollmentSubmission:
    """Input payload from the GUI form."""

    identity_id: str
    display_name: str
    role: str | None = None
    sample_files: tuple[Path, ...] = ()


class EnrollmentRegistrationService:
    """Write GUI submissions into the local enrollment store."""

    def __init__(self, store: EnrollmentStore) -> None:
        self._store = store

    @property
    def store(self) -> EnrollmentStore:
        return self._store

    def list_records(self) -> list[EnrollmentRecord]:
        return self._store.load().records

    def enroll(self, submission: EnrollmentSubmission) -> EnrollmentRecord:
        self._validate_submission(submission)
        self._store.ensure_structure()

        manifest = self._store.load()
        record = self._write_record(manifest, submission)
        self._store.save(manifest)
        return record

    def delete(self, identity_id: str) -> bool:
        identity_id = identity_id.strip()
        if not identity_id:
            raise ValueError("identity_id is required")

        manifest = self._store.load()
        records = list(manifest.records)
        remaining = [record for record in records if record.identity_id != identity_id]
        if len(remaining) == len(records):
            return False

        manifest.records[:] = remaining
        self._store.save(manifest)

        identity_dir = self._store.identity_dir(identity_id)
        if identity_dir.exists():
            shutil.rmtree(identity_dir)
        return True

    def _write_record(
        self,
        manifest: EnrollmentManifest,
        submission: EnrollmentSubmission,
    ) -> EnrollmentRecord:
        identity_dir = self._store.identity_dir(submission.identity_id)
        identity_dir.mkdir(parents=True, exist_ok=True)

        sample_images: list[str] = []
        for sample_file in submission.sample_files:
            if not sample_file.exists():
                raise FileNotFoundError(sample_file)
            destination = _unique_destination(identity_dir, sample_file.name)
            shutil.copy2(sample_file, destination)
            sample_images.append(
                f"{ENROLLMENT_IMAGES_DIR}/{submission.identity_id}/{destination.name}"
            )

        record = EnrollmentRecord(
            identity_id=submission.identity_id,
            display_name=submission.display_name,
            role=submission.role,
            sample_images=sample_images,
            enrolled_at=_utc_now(),
        )

        records = [
            existing
            for existing in manifest.records
            if existing.identity_id != submission.identity_id
        ]
        records.append(record)
        manifest.records[:] = records
        return record

    @staticmethod
    def _validate_submission(submission: EnrollmentSubmission) -> None:
        if not submission.identity_id.strip():
            raise ValueError("identity_id is required")
        if not submission.display_name.strip():
            raise ValueError("display_name is required")
        if not submission.sample_files:
            raise ValueError("at least one sample image is required")


def _load_qt_widgets():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
        QVBoxLayout,
        QWidget,
    )

    return {
        "QAbstractItemView": QAbstractItemView,
        "QFileDialog": QFileDialog,
        "QFormLayout": QFormLayout,
        "QFrame": QFrame,
        "QGridLayout": QGridLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QListWidget": QListWidget,
        "QListWidgetItem": QListWidgetItem,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSizePolicy": QSizePolicy,
        "QSpacerItem": QSpacerItem,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "Qt": Qt,
    }


def build_enrollment_window(store_root: str | Path | None = None) -> "EnrollmentWindow":
    resolved_root = default_enrollment_root() if store_root is None else store_root
    return EnrollmentWindow(EnrollmentRegistrationService(EnrollmentStore(resolved_root)))


class EnrollmentWindow(_load_qt_widgets()["QWidget"]):
    """Chinese enrollment window for family identities."""

    def __init__(self, service: EnrollmentRegistrationService) -> None:
        widgets = _load_qt_widgets()
        super().__init__()
        self._service = service
        self._sample_files: list[Path] = []

        self.setObjectName("EnrollmentWindow")
        self.setWindowTitle("家庭机器人 · 身份登记")
        self.resize(1180, 760)
        self.setMinimumSize(1024, 680)

        self.identity_id_edit = widgets["QLineEdit"]()
        self.display_name_edit = widgets["QLineEdit"]()
        self.role_edit = widgets["QLineEdit"]()
        self.sample_list = widgets["QListWidget"]()
        self.enrolled_list = widgets["QListWidget"]()
        self.status_label = widgets["QLabel"]("就绪")
        self.member_count_label = widgets["QLabel"]("0")
        self.sample_count_label = widgets["QLabel"]("0")
        self.store_path_label = widgets["QLabel"](self._format_store_path(self._service.store.root))

        self._build_ui(widgets)
        self.refresh_enrolled()

    def _build_ui(self, widgets: dict[str, object]) -> None:
        self.identity_id_edit.setPlaceholderText("例如：father / mother / child_1")
        self.display_name_edit.setPlaceholderText("例如：爸爸 / 妈妈 / 小明")
        self.role_edit.setPlaceholderText("例如：父亲 / 母亲 / 儿子")

        self.identity_id_edit.setClearButtonEnabled(True)
        self.display_name_edit.setClearButtonEnabled(True)
        self.role_edit.setClearButtonEnabled(True)
        self.sample_list.setSelectionMode(widgets["QAbstractItemView"].ExtendedSelection)
        self.sample_list.setAlternatingRowColors(True)
        self.enrolled_list.setSelectionMode(widgets["QAbstractItemView"].SingleSelection)
        self.enrolled_list.setAlternatingRowColors(True)
        self.store_path_label.setWordWrap(True)

        header = widgets["QFrame"]()
        header.setObjectName("HeroPanel")
        header_layout = widgets["QVBoxLayout"]()
        header_layout.setContentsMargins(18, 18, 18, 18)
        header_layout.setSpacing(8)

        title = widgets["QLabel"]("家庭机器人 · 身份登记中心")
        title.setObjectName("HeroTitle")
        subtitle = widgets["QLabel"](
            "为家庭成员建立长期身份档案，后续识别、重识别和问候都从这里开始。"
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)

        example = widgets["QLabel"](_format_family_examples())
        example.setObjectName("HeroExample")
        example.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(example)
        header.setLayout(header_layout)

        stats_row = widgets["QHBoxLayout"]()
        stats_row.setSpacing(12)
        stats_row.addWidget(self._build_stat_card(widgets, "已登记成员", self.member_count_label))
        stats_row.addWidget(self._build_stat_card(widgets, "当前样本", self.sample_count_label))
        stats_row.addWidget(
            self._build_stat_card(widgets, "存储位置", self.store_path_label, stretch=True)
        )

        form_layout = widgets["QFormLayout"]()
        form_layout.setLabelAlignment(widgets["Qt"].AlignRight | widgets["Qt"].AlignVCenter)
        form_layout.setFormAlignment(widgets["Qt"].AlignLeft | widgets["Qt"].AlignVCenter)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(12)
        form_layout.addRow("身份编号（必填）", self.identity_id_edit)
        form_layout.addRow("显示名称（必填）", self.display_name_edit)
        form_layout.addRow("家庭角色", self.role_edit)

        tip = widgets["QLabel"](
            "身份编号是系统内部唯一键，显示名称用于界面展示，家庭角色用于后续问候和权限区分。"
        )
        tip.setWordWrap(True)
        tip.setObjectName("HelperText")

        sample_tip = widgets["QLabel"](
            "建议添加 3 至 10 张样本，包含正脸、侧脸、不同光线和表情，能明显提升后续识别稳定性。"
        )
        sample_tip.setWordWrap(True)
        sample_tip.setObjectName("HelperText")

        member_tip = widgets["QLabel"]("这里会显示本地登记库中的成员，包含显示名称、身份编号、角色和样本数量。")
        member_tip.setWordWrap(True)
        member_tip.setObjectName("HelperText")

        add_button = widgets["QPushButton"]("添加样本")
        remove_button = widgets["QPushButton"]("删除所选")
        clear_button = widgets["QPushButton"]("清空表单")
        refresh_button = widgets["QPushButton"]("刷新名单")
        save_button = widgets["QPushButton"]("保存登记")
        save_button.setObjectName("PrimaryButton")

        add_button.clicked.connect(self._add_samples)  # type: ignore[attr-defined]
        remove_button.clicked.connect(self._remove_selected)  # type: ignore[attr-defined]
        clear_button.clicked.connect(self._clear_form)  # type: ignore[attr-defined]
        refresh_button.clicked.connect(self.refresh_enrolled)  # type: ignore[attr-defined]
        save_button.clicked.connect(self._save)  # type: ignore[attr-defined]

        button_row = widgets["QHBoxLayout"]()
        button_row.setSpacing(10)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        button_row.addWidget(refresh_button)
        button_row.addWidget(save_button)

        form_box = widgets["QGroupBox"]("登记信息")
        form_box_layout = widgets["QVBoxLayout"]()
        form_box_layout.setContentsMargins(16, 18, 16, 16)
        form_box_layout.setSpacing(12)
        form_box_layout.addLayout(form_layout)
        form_box_layout.addWidget(tip)
        form_box_layout.addLayout(button_row)
        form_box_layout.addWidget(self.status_label)
        form_box.setLayout(form_box_layout)

        sample_box = widgets["QGroupBox"]("当前样本")
        sample_box_layout = widgets["QVBoxLayout"]()
        sample_box_layout.setContentsMargins(16, 18, 16, 16)
        sample_box_layout.setSpacing(12)
        sample_box_layout.addWidget(sample_tip)
        sample_box_layout.addWidget(self.sample_list)
        sample_box.setLayout(sample_box_layout)

        enrolled_box = widgets["QGroupBox"]("已登记成员")
        enrolled_box_layout = widgets["QVBoxLayout"]()
        enrolled_box_layout.setContentsMargins(16, 18, 16, 16)
        enrolled_box_layout.setSpacing(12)
        enrolled_box_layout.addWidget(member_tip)
        enrolled_box_layout.addWidget(self.enrolled_list)
        delete_member_button = widgets["QPushButton"]("删除成员")
        delete_member_button.clicked.connect(self._delete_selected_member)  # type: ignore[attr-defined]
        enrolled_box_layout.addWidget(delete_member_button)
        enrolled_box.setLayout(enrolled_box_layout)

        right_panel = widgets["QVBoxLayout"]()
        right_panel.setSpacing(12)
        right_panel.addWidget(sample_box, 1)
        right_panel.addWidget(enrolled_box, 1)

        content_row = widgets["QHBoxLayout"]()
        content_row.setSpacing(14)
        content_row.addWidget(form_box, 5)
        content_row.addLayout(right_panel, 6)

        footer = widgets["QFrame"]()
        footer.setObjectName("FooterPanel")
        footer_layout = widgets["QHBoxLayout"]()
        footer_layout.setContentsMargins(14, 10, 14, 10)
        footer_layout.addWidget(widgets["QLabel"]("提示：登记完成后，后续摄像头识别会直接使用这份本地身份库。"))
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.status_label)
        footer.setLayout(footer_layout)

        main_layout = widgets["QVBoxLayout"]()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)
        main_layout.addWidget(header)
        main_layout.addLayout(stats_row)
        main_layout.addLayout(content_row, 1)
        main_layout.addWidget(footer)
        self.setLayout(main_layout)

        self.setStyleSheet(
            """
            QWidget#EnrollmentWindow {
                background: #f4f7fb;
                color: #0f172a;
                font-size: 14px;
            }
            QFrame#HeroPanel, QFrame#FooterPanel, QFrame#StatCard {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
            }
            QLabel#HeroTitle {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#HeroSubtitle {
                font-size: 14px;
                color: #334155;
            }
            QLabel#HeroExample, QLabel#HelperText {
                color: #475569;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
                margin-top: 18px;
                padding-top: 14px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #0f172a;
            }
            QLineEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QLineEdit:focus, QListWidget:focus {
                border-color: #2563eb;
            }
            QPushButton {
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 14px;
                color: #0f172a;
            }
            QPushButton:hover {
                background: #dbe7f5;
            }
            QPushButton#PrimaryButton {
                background: #2563eb;
                border-color: #1d4ed8;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#PrimaryButton:hover {
                background: #1d4ed8;
            }
            QLabel#StatValue {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#StatTitle {
                color: #64748b;
            }
            QLabel#StatHint {
                color: #475569;
            }
            """
        )

    def refresh_enrolled(self) -> None:
        self.enrolled_list.clear()
        records = self._service.list_records()
        if not records:
            widgets = _load_qt_widgets()
            item = widgets["QListWidgetItem"]("暂无登记成员")
            item.setFlags(item.flags() & ~widgets["Qt"].ItemIsSelectable)
            self.enrolled_list.addItem(item)
            self.member_count_label.setText("0")
            self._update_store_path_label()
            return

        widgets = _load_qt_widgets()
        for record in records:
            text = self._format_record(record)
            item = widgets["QListWidgetItem"](text)
            item.setData(widgets["Qt"].UserRole, record.identity_id)
            self.enrolled_list.addItem(item)
        self.member_count_label.setText(str(len(records)))
        self._refresh_sample_count()
        self._update_store_path_label()

    def _format_record(self, record: EnrollmentRecord) -> str:
        role = record.role or "未填写"
        return (
            f"{record.display_name}  |  编号: {record.identity_id}  |  "
            f"角色: {role}  |  样本: {len(record.sample_images)}"
        )

    @staticmethod
    def _format_store_path(root: Path) -> str:
        try:
            relative = root.relative_to(Path(__file__).resolve().parents[2])
        except ValueError:
            return str(root)
        return str(relative)

    def _build_stat_card(
        self,
        widgets: dict[str, object],
        title: str,
        value_label: "object",
        *,
        stretch: bool = False,
    ) -> "object":
        frame = widgets["QFrame"]()
        frame.setObjectName("StatCard")
        frame_layout = widgets["QVBoxLayout"]()
        frame_layout.setContentsMargins(16, 14, 16, 14)
        frame_layout.setSpacing(4)
        title_label = widgets["QLabel"](title)
        title_label.setObjectName("StatTitle")
        value_label.setObjectName("StatValue")
        hint = widgets["QLabel"]("本地登记库")
        hint.setObjectName("StatHint")
        frame_layout.addWidget(title_label)
        frame_layout.addWidget(value_label)
        frame_layout.addWidget(hint)
        frame.setLayout(frame_layout)
        if stretch:
            frame.setSizePolicy(widgets["QSizePolicy"].Expanding, widgets["QSizePolicy"].Preferred)
        return frame

    def _refresh_sample_count(self) -> None:
        self.sample_count_label.setText(str(len(self._sample_files)))

    def _update_store_path_label(self) -> None:
        self.store_path_label.setText(self._format_store_path(self._service.store.root))

    def _add_samples(self) -> None:
        widgets = _load_qt_widgets()
        file_paths, _ = widgets["QFileDialog"].getOpenFileNames(
            self,
            "选择样本图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_paths:
            return

        for file_path in file_paths:
            path = Path(file_path)
            if path not in self._sample_files:
                self._sample_files.append(path)
                self.sample_list.addItem(path.name)
        self._refresh_sample_count()
        self._set_status(f"已选择 {len(self._sample_files)} 张样本")

    def _remove_selected(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.sample_list.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        for row in selected_rows:
            item = self.sample_list.takeItem(row)
            if item is None:
                continue
            self._sample_files.pop(row)
        self._refresh_sample_count()
        self._set_status(f"已选择 {len(self._sample_files)} 张样本")

    def _delete_selected_member(self) -> None:
        widgets = _load_qt_widgets()
        selected_items = self.enrolled_list.selectedItems()
        if not selected_items:
            widgets["QMessageBox"].information(self, "删除成员", "请先在已登记成员列表中选择一个成员。")
            return

        item = selected_items[0]
        identity_id = item.data(widgets["Qt"].UserRole)
        if not identity_id:
            widgets["QMessageBox"].warning(self, "删除成员", "无法读取所选成员的身份编号。")
            return

        display_text = item.text()
        answer = widgets["QMessageBox"].question(
            self,
            "确认删除",
            f"确定要删除该登记成员吗？\n\n{display_text}\n\n删除后将同时移除本地样本和登记记录。",
            widgets["QMessageBox"].Yes | widgets["QMessageBox"].No,
            widgets["QMessageBox"].No,
        )
        if answer != widgets["QMessageBox"].Yes:
            return

        deleted = self._service.delete(str(identity_id))
        if not deleted:
            widgets["QMessageBox"].information(self, "删除成员", "未找到对应成员，可能已被删除。")
            self.refresh_enrolled()
            return

        self.refresh_enrolled()
        self._set_status(f"已删除：{display_text}")
        widgets["QMessageBox"].information(self, "删除成功", f"已删除：{display_text}")

    def _clear_form(self) -> None:
        self._sample_files.clear()
        self.sample_list.clear()
        self.identity_id_edit.clear()
        self.display_name_edit.clear()
        self.role_edit.clear()
        self._refresh_sample_count()
        self._set_status("已清空表单")

    def _save(self) -> None:
        widgets = _load_qt_widgets()
        try:
            record = self._service.enroll(
                EnrollmentSubmission(
                    identity_id=self.identity_id_edit.text().strip(),
                    display_name=self.display_name_edit.text().strip(),
                    role=self.role_edit.text().strip() or None,
                    sample_files=tuple(self._sample_files),
                )
            )
        except Exception as exc:  # pragma: no cover - UI feedback path
            widgets["QMessageBox"].critical(self, "登记失败", str(exc))
            return

        self._clear_form()
        self.refresh_enrolled()
        self._set_status(
            f"已保存：{record.display_name}，共 {len(record.sample_images)} 张样本"
        )
        widgets["QMessageBox"].information(self, "登记成功", record.display_name)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
